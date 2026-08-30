// ImageCompression：IMGIC/PVRIC image-compression block 的 profile-selectable
// module boundary。PVRIC = PowerVR Image Compression；IMGIC 的完整展開不在此
// 臆造。PBE 目前輸出未壓縮 RGBA8，不能視為已有 compression codec。
// 現階段為無 ports/process/timing、未連線的 elaboration placeholder。
#include "compression/image_compression.h"

namespace pvrgpu::stub {

ImageCompression::ImageCompression(sc_core::sc_module_name name)
    : sc_module(name) {}

}  // namespace pvrgpu::stub
