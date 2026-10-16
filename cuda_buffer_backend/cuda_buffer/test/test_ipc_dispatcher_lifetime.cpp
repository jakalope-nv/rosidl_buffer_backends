// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <filesystem>
#include <iterator>
#include <memory>
#include <thread>
#include <vector>

#include "cuda_buffer/cuda_buffer_ipc_manager.hpp"
#include "gtest/gtest.h"

namespace
{
size_t thread_count()
{
  return std::distance(std::filesystem::directory_iterator("/proc/self/task"),
    std::filesystem::directory_iterator());
}

bool wait_for_thread_count(size_t expected)
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  do {
    if (thread_count() == expected) {return true;}
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < deadline);
  return false;
}
}  // namespace

TEST(CudaIPCDispatcher, SharedWorkerStopsWithLastManagerAndCanRestart)
{
  const size_t original = thread_count();
  for (int cycle = 0; cycle < 3; ++cycle) {
    std::vector<std::unique_ptr<cuda_buffer_backend::CudaVmmIPCManager>> managers;
    for (int i = 0; i < 8; ++i) {
      managers.push_back(std::make_unique<cuda_buffer_backend::CudaVmmIPCManager>());
    }
    ASSERT_TRUE(wait_for_thread_count(original + 1));
    managers.resize(1);
    EXPECT_EQ(thread_count(), original + 1);
    managers.clear();
    ASSERT_TRUE(wait_for_thread_count(original)) << "IPC worker survived its last owner";
  }
}
