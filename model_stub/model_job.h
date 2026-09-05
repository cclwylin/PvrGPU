// Module：ModelJob（非縮寫，意為一次送進常駐模型的工作）。
// 功能：描述一次 flush 的輸入與輸出。SystemC 每個 process 只 elaborate 一
// 次，因此 module 與 FIFO 必須比單一次呼叫活得更久；工作本身改由這個
// slot 傳入，Submitter 與 JsonReporter 在 `start` event 上被喚醒後各自認領。
#pragma once

#include "model_types.h"

#include <systemc>

#include <cstdint>
#include <string>
#include <vector>

namespace pvrgpu::stub {

/*
 * One flush of work through the persistent model.
 *
 * SystemC elaborates once per process, so the model cannot be rebuilt for
 * every submission the driver makes.  The modules therefore outlive any one
 * flush and the work reaches them through this slot instead of through their
 * constructors: the session fills in `options`, notifies `start`, and runs
 * `sc_start()` until `complete` comes back.  Between flushes both processes
 * sit in `wait(start)`, which leaves the pipeline idle rather than finished --
 * which is what the hardware does, and what lets a second draw follow a
 * readback.
 */
struct ModelJob {
  // The job's configuration, adopted by Submitter and JsonReporter the moment
  // they wake.  It is written only while no process is running.
  Options options;

  sc_core::sc_event start{"pvrgpu_model_job_start"};

  // True from the moment the session notifies `start` until it has collected
  // the result.  A spurious wake with this clear is ignored.
  bool running = false;
  // The submitter has queued every submission this job describes.
  bool submitted = false;
  // The reporter has published the job's last record.  This is what the
  // session waits for.
  bool complete = false;
  bool failed = false;
  std::string error;

  // The job's last physical DRAM readback, as RGBA8.  This is what a
  // `glReadPixels` on a colour attachment ends up copying.
  std::vector<std::uint8_t> framebuffer;
  std::uint32_t framebuffer_width = 0;
  std::uint32_t framebuffer_height = 0;

  void Begin(const Options &job_options) {
    options = job_options;
    running = true;
    submitted = false;
    complete = false;
    failed = false;
    error.clear();
    framebuffer.clear();
    framebuffer_width = 0;
    framebuffer_height = 0;
  }

  // Records the first failure only: a later stage failing because an earlier
  // one did should not overwrite the cause.
  void Fail(const std::string &message) {
    if (failed)
      return;
    failed = true;
    error = message;
  }

  void PublishFramebuffer(const std::vector<std::uint8_t> &pixels,
                          std::uint32_t width, std::uint32_t height) {
    if (static_cast<std::uint64_t>(pixels.size()) !=
        static_cast<std::uint64_t>(width) * height * 4U) {
      return;
    }
    framebuffer = pixels;
    framebuffer_width = width;
    framebuffer_height = height;
  }
};

} // namespace pvrgpu::stub
