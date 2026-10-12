#pragma once
#include <cassert>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// A small sparse-set ECS.
//
//  - An Entity is just an integer ID.
//  - Each component type lives in its own Pool<T>: a tightly packed array of T
//    (the "dense" array), plus a lookup table from entity ID to dense index
//    (the "sparse" array). Iterating a pool walks contiguous memory.
//  - Systems are plain functions that iterate pools via Registry::each.
//
// Known limits (deliberate, for now): no generation counter on entity IDs, so a
// stale handle to a destroyed entity can alias a newer one; and components must
// not be added/removed from the pools being iterated inside an each() callback.
// ---------------------------------------------------------------------------

using Entity = uint32_t;
inline constexpr Entity NULL_ENTITY = 0xFFFFFFFFu;

struct IPool {
    virtual ~IPool() = default;
    virtual void remove(Entity e) = 0;
};

template <typename T>
class Pool final : public IPool {
public:
    bool has(Entity e) const { return e < sparse_.size() && sparse_[e] != NONE; }

    // The returned reference is valid until the next add() or remove() on this pool.
    T& add(Entity e, T value = T{}) {
        if (e >= sparse_.size()) sparse_.resize((size_t)e + 1, NONE);
        if (sparse_[e] != NONE) {                    // already has one: overwrite
            dense_[sparse_[e]] = std::move(value);
            return dense_[sparse_[e]];
        }
        sparse_[e] = (uint32_t)dense_.size();
        dense_.push_back(std::move(value));
        entities_.push_back(e);
        return dense_.back();
    }

    // One lookup instead of has() + get(): nullptr if the entity has no such component.
    T* find(Entity e) {
        if (e >= sparse_.size()) return nullptr;
        uint32_t idx = sparse_[e];
        return idx == NONE ? nullptr : &dense_[idx];
    }

    T&       get(Entity e)       { return dense_[sparse_[e]]; }
    const T& get(Entity e) const { return dense_[sparse_[e]]; }

    // Swap-and-pop: keeps the dense array packed. Order of the remaining items may change.
    void remove(Entity e) override {
        if (!has(e)) return;
        uint32_t idx = sparse_[e];
        Entity lastEntity = entities_.back();
        dense_[idx]    = std::move(dense_.back());
        entities_[idx] = lastEntity;
        sparse_[lastEntity] = idx;
        dense_.pop_back();
        entities_.pop_back();
        sparse_[e] = NONE;
    }

    size_t size() const { return dense_.size(); }
    std::vector<T>&            data()           { return dense_; }
    const std::vector<Entity>& entities() const { return entities_; }

private:
    static constexpr uint32_t NONE = 0xFFFFFFFFu;
    std::vector<T>        dense_;      // the components, packed
    std::vector<Entity>   entities_;   // entities_[i] owns dense_[i]
    std::vector<uint32_t> sparse_;     // entity ID -> index into dense_, or NONE
};

class Registry {
public:
    // ---- entities ----
    Entity create() {
        if (!free_.empty()) {
            Entity e = free_.back();
            free_.pop_back();
            alive_[e] = 1;
            return e;
        }
        alive_.push_back(1);
        return (Entity)(alive_.size() - 1);
    }

    void destroy(Entity e) {
        if (!alive(e)) return;
        for (auto& p : pools_) if (p) p->remove(e);
        alive_[e] = 0;
        free_.push_back(e);
    }

    bool   alive(Entity e) const { return e < alive_.size() && alive_[e]; }
    size_t size() const { return alive_.size() - free_.size(); }

    // ---- components ----
    template <typename T>
    T& add(Entity e, T value = T{}) {
        assert(alive(e));
        return pool<T>().add(e, std::move(value));
    }

    template <typename T>
    bool has(Entity e) const {
        const Pool<T>* p = poolIfExists<T>();
        return p && p->has(e);
    }

    template <typename T>
    T& get(Entity e) { return pool<T>().get(e); }

    template <typename T>
    void remove(Entity e) {
        if (Pool<T>* p = poolIfExists<T>()) p->remove(e);
    }

    template <typename T>
    Pool<T>& pool() {
        uint32_t id = typeId<T>();
        if (id >= pools_.size()) pools_.resize((size_t)id + 1);
        if (!pools_[id]) pools_[id] = std::make_unique<Pool<T>>();
        return static_cast<Pool<T>&>(*pools_[id]);
    }

    // Calls fn(entity, First&, Rest&...) for every entity that has ALL the listed components.
    // Iterates the pool of the FIRST type, so list the rarest component first.
    //     reg.each<Spin, Transform>([&](Entity, Spin& s, Transform& t) { ... });
    template <typename First, typename... Rest, typename Fn>
    void each(Fn&& fn) {
        Pool<First>& lead = pool<First>();
        std::tuple<Pool<Rest>&...> others{ pool<Rest>()... };
        std::vector<First>&        data = lead.data();
        const std::vector<Entity>& ents = lead.entities();
        for (size_t i = 0; i < data.size(); ++i) {
            Entity e = ents[i];
            if constexpr (sizeof...(Rest) == 0) {
                fn(e, data[i]);
            } else {
                std::tuple<Rest*...> found{ std::get<Pool<Rest>&>(others).find(e)... };
                if (!((std::get<Rest*>(found) != nullptr) && ...)) continue;
                fn(e, data[i], *std::get<Rest*>(found)...);
            }
        }
    }

private:
    template <typename T>
    Pool<T>* poolIfExists() const {
        uint32_t id = typeId<T>();
        if (id >= pools_.size() || !pools_[id]) return nullptr;
        return static_cast<Pool<T>*>(pools_[id].get());
    }

    static uint32_t& typeCounter() { static uint32_t c = 0; return c; }
    template <typename T>
    static uint32_t typeId() { static const uint32_t id = typeCounter()++; return id; }

    std::vector<uint8_t>               alive_;
    std::vector<Entity>                free_;
    std::vector<std::unique_ptr<IPool>> pools_;
};
