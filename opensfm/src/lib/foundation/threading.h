#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace foundation {

template <typename T>
class ConcurrentQueue {
 public:
  void Push(T result) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(result));
    cv_.notify_one();
  }

  bool Pop(T& result) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty() || finished_; });
    if (queue_.empty()) {
      return false;
    }
    result = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  void Finish() {
    std::lock_guard<std::mutex> lock(mutex_);
    finished_ = true;
    cv_.notify_all();
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool finished_ = false;
};

}  // namespace foundation
