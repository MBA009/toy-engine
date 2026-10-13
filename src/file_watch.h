#pragma once
#include <filesystem>
#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Polls file modification times (Step 6.4 hot reload). Call poll() every so often (say twice a second);
// it returns the files that changed since the last time they were seen.
//
// A change is only reported once the timestamp has stayed the same for one full poll. Editors and
// compilers write a file in several steps, so reacting to the first timestamp change would often mean
// reading a half-written file.
// ---------------------------------------------------------------------------
class FileWatcher {
public:
    // Start watching; the file's current state is the baseline (so existing files don't report as "changed").
    // A file that doesn't exist yet is fine: it reports once it appears.
    void watch(const std::string& path) {
        if (entries_.count(path)) return;
        Entry e;
        e.seen = e.pending = stamp(path);
        entries_[path] = e;
    }
    void unwatch(const std::string& path) { entries_.erase(path); }
    void clear() { entries_.clear(); }
    bool watching(const std::string& path) const { return entries_.count(path) != 0; }

    // "I already know about the file's current state" (e.g. we just wrote it ourselves): no report for it
    void acknowledge(const std::string& path) {
        auto it = entries_.find(path);
        if (it != entries_.end()) it->second.seen = it->second.pending = stamp(path);
    }

    // Files that changed and have settled. A file that was deleted is not reported (there is nothing to
    // reload); if it comes back, that is a change.
    std::vector<std::string> poll() {
        std::vector<std::string> changed;
        for (auto& [path, e] : entries_) {
            Stamp now = stamp(path);
            if (now == e.seen) { e.pending = e.seen; continue; }    // unchanged (or changed and changed back)
            if (!(now == e.pending)) { e.pending = now; continue; } // still moving: wait for the next poll
            e.seen = now;                                           // settled
            if (now.exists) changed.push_back(path);
        }
        return changed;
    }

private:
    struct Stamp {
        bool exists = false;
        std::filesystem::file_time_type time{};
        std::uintmax_t size = 0;
        bool operator==(const Stamp& o) const { return exists == o.exists && time == o.time && size == o.size; }
    };
    struct Entry { Stamp seen, pending; };

    static Stamp stamp(const std::string& path) {
        Stamp s;
        std::error_code ec;
        auto t = std::filesystem::last_write_time(path, ec);
        if (ec) return s;
        s.exists = true;
        s.time = t;
        s.size = std::filesystem::file_size(path, ec);   // catches same-timestamp rewrites on coarse filesystems
        if (ec) s.size = 0;
        return s;
    }

    std::map<std::string, Entry> entries_;
};
