#pragma once
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "components.h"
#include "ecs.h"

// ---------------------------------------------------------------------------
// A minimal behavior tree (Step 7.4).
//
//  - A tree is made of nodes. Each tick, a node returns Success, Failure or Running.
//  - Sequence: runs its children in order; stops at the first that is not Success ("do all of these").
//  - Selector: runs its children in order; stops at the first that is not Failure ("do the first that works").
//  - Leaf: a function that checks something or does something.
//  - The whole tree is re-evaluated from the root every tick (no "remember where I was" state), so a
//    higher-priority branch can interrupt a lower one the moment its condition becomes true. Leaves that act
//    therefore have to be safe to run every frame: they only issue a new path when something changed.
//  - The tree holds no per-agent data. Everything an agent remembers lives in its AIBrain component.
// ---------------------------------------------------------------------------

enum class BTStatus : int8_t { Success = 0, Failure = 1, Running = 2 };

inline const char* toString(BTStatus s) {
    return s == BTStatus::Success ? "Success" : (s == BTStatus::Failure ? "Failure" : "Running");
}

// Finds a walkable route between two feet positions (the navmesh, in the editor; a fake one in tests)
using PathFn = std::function<bool(const Vec3& from, const Vec3& to, std::vector<Vec3>& corners)>;

// Everything a leaf may look at or change, for one agent on one tick
struct BTContext {
    Registry& world;
    Entity self;
    Entity player;               // NULL_ENTITY if there is no player
    float dt;
    const PathFn& findPath;
    AIAgent& agent;
    Transform& transform;
    AIBrain& brain;
    Health* health;              // null if the agent has no Health

    Vec3 feet() const { return { transform.position.x, transform.position.y - agent.footOffset, transform.position.z }; }
    bool hasPlayer() const { return player != NULL_ENTITY; }
    // Where the player's feet are
    Vec3 playerFeet() const {
        const Transform& t = world.get<Transform>(player);
        float foot = world.has<AIAgent>(player) ? world.get<AIAgent>(player).footOffset : 0.0f;
        return { t.position.x, t.position.y - foot, t.position.z };
    }
};

struct BTNode {
    enum class Kind { Selector, Sequence, Leaf };
    Kind kind = Kind::Leaf;
    std::string name;
    int id = -1;                                             // numbered depth-first by BehaviorTree
    std::vector<std::unique_ptr<BTNode>> children;
    std::function<BTStatus(BTContext&)> action;              // leaves only
};
using BTNodePtr = std::unique_ptr<BTNode>;

inline BTNodePtr leaf(std::string name, std::function<BTStatus(BTContext&)> fn) {
    auto n = std::make_unique<BTNode>();
    n->kind = BTNode::Kind::Leaf;
    n->name = std::move(name);
    n->action = std::move(fn);
    return n;
}

namespace detail {
inline void addChildren(BTNode&) {}
template <typename First, typename... Rest>
void addChildren(BTNode& parent, First&& first, Rest&&... rest) {
    parent.children.push_back(std::move(first));
    addChildren(parent, std::forward<Rest>(rest)...);
}
}  // namespace detail

template <typename... Children>
BTNodePtr selector(std::string name, Children&&... children) {
    auto n = std::make_unique<BTNode>();
    n->kind = BTNode::Kind::Selector;
    n->name = std::move(name);
    detail::addChildren(*n, std::forward<Children>(children)...);
    return n;
}

template <typename... Children>
BTNodePtr sequence(std::string name, Children&&... children) {
    auto n = std::make_unique<BTNode>();
    n->kind = BTNode::Kind::Sequence;
    n->name = std::move(name);
    detail::addChildren(*n, std::forward<Children>(children)...);
    return n;
}

class BehaviorTree {
public:
    explicit BehaviorTree(BTNodePtr root) : root_(std::move(root)) { index(*root_); }
    const BTNode& root() const { return *root_; }
    int nodeCount() const { return (int)nodes_.size(); }
    const BTNode& node(int id) const { return *nodes_[(size_t)id]; }

private:
    void index(BTNode& n) {
        n.id = (int)nodes_.size();
        nodes_.push_back(&n);
        for (auto& c : n.children) index(*c);
    }
    BTNodePtr root_;
    std::vector<const BTNode*> nodes_;
};

// Runs one node (and whatever it calls) and records what each visited node returned in brain.trace
inline BTStatus tickNode(const BTNode& n, BTContext& c) {
    BTStatus s = BTStatus::Failure;
    switch (n.kind) {
        case BTNode::Kind::Leaf:
            s = n.action(c);
            c.brain.lastLeaf = n.id;
            break;
        case BTNode::Kind::Selector:
            for (const auto& child : n.children) {
                s = tickNode(*child, c);
                if (s != BTStatus::Failure) break;
            }
            break;
        case BTNode::Kind::Sequence:
            s = BTStatus::Success;
            for (const auto& child : n.children) {
                s = tickNode(*child, c);
                if (s != BTStatus::Success) break;
            }
            break;
    }
    if ((size_t)n.id < c.brain.trace.size()) c.brain.trace[(size_t)n.id] = (int8_t)s;
    return s;
}

// ---------------------------------------------------------------------------
// The leaves
// ---------------------------------------------------------------------------
namespace bt {

inline float dist2D(const Vec3& a, const Vec3& b) {
    const float dx = a.x - b.x, dz = a.z - b.z;
    return std::sqrt(dx * dx + dz * dz);
}

// Gives up the path a different behaviour was following, so the new one starts clean
inline void takeOver(BTContext& c, uint8_t owner) {
    if (c.brain.pathOwner == owner) return;
    c.agent.stop();
    c.brain.pathOwner = owner;
    c.brain.repathTimer = 0.0f;
    c.brain.patrolStarted = false;
    c.brain.fleeTarget = -1;
}

inline bool issuePath(BTContext& c, const Vec3& target) {
    std::vector<Vec3> corners;
    if (!c.findPath || !c.findPath(c.feet(), target, corners) || corners.empty()) return false;
    c.agent.setPath(std::move(corners));
    return true;
}

// Condition: the player is close enough to notice. Once noticed, it takes a bigger distance to lose them.
inline BTStatus isPlayerInRange(BTContext& c) {
    AIBrain& b = c.brain;
    if (!c.hasPlayer()) { b.chasing = false; return BTStatus::Failure; }
    const float range = b.chasing ? b.detectRange * 1.3f : b.detectRange;
    b.chasing = dist2D(c.feet(), c.playerFeet()) <= range;
    return b.chasing ? BTStatus::Success : BTStatus::Failure;
}

// Condition: health is low. Flips on below fleeBelow and stays on until recoverAt, so it does not flicker.
inline BTStatus isHealthLow(BTContext& c) {
    AIBrain& b = c.brain;
    if (!c.health) return BTStatus::Failure;
    const float f = c.health->fraction();
    if (b.fleeing) { if (f >= b.recoverAt) b.fleeing = false; }
    else if (f < b.fleeBelow) b.fleeing = true;
    return b.fleeing ? BTStatus::Success : BTStatus::Failure;
}

// Action: walk to the player, re-aiming a few times a second since they move. Stops when touching them.
inline BTStatus moveToPlayer(BTContext& c) {
    AIBrain& b = c.brain;
    if (!c.hasPlayer()) return BTStatus::Failure;
    takeOver(c, AIBrain::OWNER_CHASE);
    const Vec3 target = c.playerFeet();
    if (dist2D(c.feet(), target) <= b.contactRange) { c.agent.stop(); return BTStatus::Running; }   // in contact
    b.repathTimer -= c.dt;
    if (!c.agent.walking() || b.repathTimer <= 0.0f) {
        b.repathTimer = 0.3f;
        if (!issuePath(c, target)) return BTStatus::Failure;      // no route: let the tree try something else
    }
    return BTStatus::Running;
}

// Action: run to the patrol point farthest from the player (or stay put if there are none)
inline BTStatus flee(BTContext& c) {
    AIBrain& b = c.brain;
    takeOver(c, AIBrain::OWNER_FLEE);
    if (!c.hasPlayer() || b.patrol.empty()) { c.agent.stop(); return BTStatus::Running; }   // nowhere to run: wait it out
    b.repathTimer -= c.dt;
    if (b.repathTimer > 0.0f && c.agent.walking()) return BTStatus::Running;
    b.repathTimer = 0.5f;
    const Vec3 threat = c.playerFeet();
    int best = 0;
    float bestDist = -1.0f;
    for (size_t i = 0; i < b.patrol.size(); i++) {
        const float d = dist2D(b.patrol[i], threat);
        if (d > bestDist) { bestDist = d; best = (int)i; }
    }
    const Vec3& target = b.patrol[(size_t)best];
    if (best != b.fleeTarget || !c.agent.walking()) {
        if (dist2D(c.feet(), target) > 0.15f) {
            if (!issuePath(c, target)) return BTStatus::Failure;
        }
        b.fleeTarget = best;
    }
    return BTStatus::Running;
}

// Action: walk between the patrol points in a loop, pausing at each
inline BTStatus patrol(BTContext& c) {
    AIBrain& b = c.brain;
    if (b.patrol.empty()) return BTStatus::Failure;
    takeOver(c, AIBrain::OWNER_PATROL);
    const size_t n = b.patrol.size();
    if (b.patrolIndex >= n) b.patrolIndex = 0;
    if (c.agent.walking()) return BTStatus::Running;
    if (b.waitTimer > 0.0f) { b.waitTimer -= c.dt; return BTStatus::Running; }
    if (b.patrolStarted) {                              // the walk to this point has ended: on to the next one
        b.patrolIndex = (uint32_t)((b.patrolIndex + 1) % n);
        b.patrolStarted = false;
        b.waitTimer = b.patrolWait;
        if (b.waitTimer > 0.0f) return BTStatus::Running;
    }
    if (!issuePath(c, b.patrol[b.patrolIndex])) {
        b.patrolIndex = (uint32_t)((b.patrolIndex + 1) % n);   // can't get there: skip it
        return BTStatus::Failure;
    }
    b.patrolStarted = true;
    return BTStatus::Running;
}

// Action: do nothing (the fallback when there is nothing else to do)
inline BTStatus idle(BTContext& c) {
    takeOver(c, AIBrain::OWNER_IDLE);
    return BTStatus::Running;
}

}  // namespace bt

// The standard enemy: run away when hurt; otherwise chase the player if they are close; otherwise patrol;
// otherwise stand still.
inline BehaviorTree makeCombatTree() {
    return BehaviorTree(selector("Root",
        sequence("Flee when hurt",   leaf("IsHealthLow", bt::isHealthLow),       leaf("Flee", bt::flee)),
        sequence("Chase the player", leaf("IsPlayerInRange", bt::isPlayerInRange), leaf("MoveToPlayer", bt::moveToPlayer)),
        leaf("Patrol", bt::patrol),
        leaf("Idle", bt::idle)));
}
