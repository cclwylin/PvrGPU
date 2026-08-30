# IMG D-Series DXTP-64-2048 Technical Specification

* **Document:** Technical Specification / Engineering Datasheet
* **Core Identification:** DXTP-64-2048 (73.V.3264.1011 Graphics Core)
* **Revision:** 1.34
* **Published Date:** 15/10/2024
* **Document Status:** First Draft


---

## 1. Introduction

This technical specification is an engineering datasheet covering the performance and detailed features of the programmable **IMG™ D-Series DXTP-64-2048 GPU** (identification `73.V.3264.1011`).

### 1.1 Supported Workload Types
The D-Series cores process a number of distinct workload types:
1. **3D Graphics Workload:** Involves processing vertex data and pixel data for rendering 3D scenes using a Tile-Based Deferred Rendering (TBDR) architecture.
2. **Compute Workload (GP-GPU):** Involves general-purpose data processing with fine-grained task switching and workgroup execution.
3. **2D Workload:** Involves processing pixel data for rendering 2D objects. The 2D workload is structured as a series of 2D render packets by the driver, known as **blits**.

### 1.2 System Block Diagram Overview
The top-level architecture consists of the following interconnected subsystems:
* **Host & Interconnect:**
  * SoC Bus Interface (AXI3 Slave Interface)
  * XPU Interface
  * Control and Register Bus
  * Dedicated RISC-V Firmware Processor
* **Data Masters & Sequencing:**
  * Vertex Data Master
  * Pixel Data Master
  * Compute Data Master
  * Domain Data Master
  * 2D Data Master
  * Programmable Data Sequencer (PDS)
* **Execution & Processing Units:**
  * Unified Shading Clusters (USC) – 8 USCs total
  * Texture Processing Unit (TPU) supporting ASTC, PVRTC, and uncompressed formats
  * Geometry Phase Pipeline
  * Fragment Phase Pipeline
* **Caches & Memory Infrastructure:**
  * Mixed Cache Unit (MCU)
  * Specialised Texture Cache Unit (TCU)
  * IMGIC / PVRIC Compression Modules
  * MMU (Memory Management Unit) with 40-bit virtual address space
  * System Level Cache (SLC) – 2048 KB
  * System Memory Bus Interface

---

## 2. Programming Features and Supported Standards

### 2.1 Base Architecture & API Compliance
The core is fully compliant with standard graphics, compute, and system APIs:
* **OpenGL ES:** 3.2
* **Vulkan:** 1.3
* **OpenCL:** 1.2 EP, 2.1 EP, and 3.0 EP (Embedded Profile)
* **EGL:** 1.4

### 2.2 Key Architectural Features
* **Tile-Based Deferred Rendering (TBDR):** Hardware architecture optimized for 3D graphics workloads with concurrent processing of multiple tiles.
* **Robust Buffer Access:** Hardware enforcement to prevent out-of-bounds access on defined memory resources.
* **Tile Processing Time Tracking:** Built-in hardware tracking for the 3D fragment processing phase to assist in performance debugging and profiling.
* **Anti-Aliasing:** Programmable high-quality image anti-aliasing supporting up to 8× multisampling (MSAA).
* **Fine-Grain Triangle Culling:** Early removal of non-visible geometry to reduce downstream processing.
* **DRM Security:** Hardware-level Digital Rights Management (DRM) support.
* **GPU Virtualization:**
  * Up to **8 Virtual GPUs**.
  * Supported via **IMG HyperLane Technology** (8 isolated hardware hyperlanes).
* **AI Synergy:** Supports Imagination AI Synergy when paired with an Imagination NNA (Neural Network Accelerator) core.
* **Asynchronous Fast 2D Renders:** High-throughput processing for 2D blits and composition.
* **Unified Shading Engine:** Multi-threaded Unified Shading Cluster (USC) engine combining pixel shader, vertex shader, hull/domain shaders (tessellation), and GP-GPU compute shader execution into high-efficiency SIMD ALUs.
* **Memory & Addressing:**
  * Fully virtualized memory addressing (up to 1 TB address space, 40-bit virtual address width).
  * Unified Memory Architecture (UMA) support.
* **Task & Power Management:**
  * Fine-grained task switching, workload balancing, and dynamic power management.
  * Advanced DMA-driven operation to minimize host CPU intervention.
  * Full Data Master task removal capability.
* **Caches & Memory Hierarchy:**
  * Dedicated System Level Cache (SLC).
  * Specialised Texture Cache Unit (TCU).
  * Mixed Cache Unit (MCU).
  * Compressed Texture Decoding.
* **Hardware Data Compression:**
  * **PVRGC (PowerVR Geometry Compression):** Lossless data compression executed during the geometry processing phase of 3D workloads.
  * **PVRIC (PowerVR Image Compression / FBCDC):** Lossless and/or visually lossless frame buffer compression and decompression algorithm (PVRIC version 5).
* **Dedicated Firmware Processor:**
  * Single-threaded RISC-V core dedicated to D-Series firmware execution.
  * 16 KB Instruction Cache, 2 KB Data Cache, and dedicated core memory.
* **Modular SPU Structure & Power Gating:**
  * Composed of Scalable Processing Units (SPUs) and one common system-level module (**Jones**).
  * Separate power island for each SPU.
  * Independent power islands within Jones for both **SPARROW** and **CHEST** blocks (where CHEST contains the majority of L2 Cache SRAM and logic).
  * On-chip Performance, Power, and Statistics Registers.

---

## 3. Core Engine & Functional Subsystems

### 3.1 Unified Shading Cluster (USC) Features
* **Instance Parallelism:** 128 parallel instances per clock.
* **Local Storage:** Dedicated local data and instruction caches.
* **Instruction Set:** Variable-length instruction set encoding.
* **Atomics:** Full support for OpenCL™ atomic operations (including compare-and-swap/exchange) and 64-bit global atomics.
* **Execution Model:** Flexible scalar and vector SIMD execution model.
* **Precision:** Native support for F16 data types in complex ALUs.
* **Addressing:** Flat addressing support.
* **Storage Separation:** Split Shared and Coefficient Stores.
* **Bindless Resources:** Native bindless image and texture support.

### 3.2 3D Graphics Features & Rasterisation
* **Deferred Pixel Shading:** Processes only visible fragments post-hidden surface removal.
* **Depth / Stencil:**
  * On-chip tile floating-point depth buffer.
  * 8-bit stencil with on-chip tile stencil buffer.
  * 32 parallel depth/stencil tests per clock for MSAA renders; 16 parallel tests per clock for non-MSAA renders.
* **ISP Integration:** One Image Synthesis Processor (ISP) per USC.
* **Context Switching:** Rapid hardware-assisted context switching.
* **Variable Rate Shading:** Fragment Shading Rate (FSR) support in pipeline mode (up to 4×4 pixel fragments).
* **Maximum Tiles in Flight:** Up to 4 tiles per ISP.

### 3.3 Texturing, Filtering & Resolution Support
* **Texture Lookups:** Supports load from source instructions.
* **Filtering Capabilities:**
  * Sample details: sample data and coefficient support.
  * Bilinear, trilinear, and volume filtering.
  * Anisotropic filtering.
  * Corner filtering support for Cube Environment Mapped textures and seamless filtering across cube faces.
  * F32 / U32 / S32 Border Colour support.
  * Chroma interpolation for YUV 420/422 formats.
* **Supported Texture Formats:**
  * PVRTC I and PVRTC II compressed texture formats.
  * ASTC LDR and ASTC HDR compressed formats.
  * BC 1 through BC 5 compressed formats.
  * ETC / EAC compressed formats.
  * Border textures.
  * PVRIC lossless and visually lossless compression support for non-compressed textures and YUV textures.
  * Texture Arrays: Up to 2K layers.
  * Buffer Textures: Up to 131,072K elements.
  * YUV Planar: 1, 2, and 3 planar formats (420 / 422 / 444) in 8-bit and 10-bit depths.
  * 10-bit sRGB and YUV format support.
* **Maximum Resolutions & Anti-Aliasing:**
  * **Max Framebuffer Resolution:** 32,768 × 32,768 (32K × 32K)
  * **Max Texture Resolution:** 32,768 × 32,768 (32K × 32K)
  * **Multisampling:** Up to 8× MSAA.

### 3.4 Primitive Assembly & Render to Buffers
* **Primitive Assembly:**
  * Early hidden object removal.
  * Vertex compression.
  * Tile acceleration.
* **Render-to-Buffer Features:**
  * Twiddled memory format support.
  * Multiple on-chip Render Targets (MRT).
  * Lossless and/or visually lossless Frame Buffer Compression / Decompression (FBCDC).
  * Programmable Geometry Shader Support.
  * Direct Geometry Stream Out (Transform Feedback) and Parallel Stream Out.

### 3.5 Compute Features (GP-GPU)
* **Compute Primitives:** 1D, 2D, and 3D compute grid primitives.
* **Workgroup Size:** Maximum workgroup size of 1024 work-items.
* **Data Movement:** Per-task input data DMA directly into the USC Unified Store.
* **Execution Control:** Conditional execution, execution fences, and barrier synchronization.
* **Workload Overlapping:** Compute workloads can overlap with graphics/2D workloads using barriers.
* **Arithmetic & Control Flow:** Round to nearest even (IEEE 754 compliance), hardware Call/Return, and pre-emption support.

### 3.6 PVRIC (FBCDC) Features
* **Compression Modes:** Lossless and/or visually lossless image compression.
* **Bandwidth Reduction:** Typically 50% internal and external compression.
* **Optimizations:** Compressed tile packing; MSAA edge, constant colour, and fast clear optimizations.
* **Scalability:** Scales with memory interface width and data rates.
* **Dimensions:** Supports 1D/2D textures and texture arrays.
* **Throughput:** High-performance out-of-order processing with **8 decompression modules**.

---

## 4. Architectural Characteristics & Storage Configurations

### 4.1 Core Configuration
* **USC Count:** 8 Unified Shading Clusters (USCs).
* **Firmware Processor:** Single-threaded RISC-V core with 16 KB I-Cache, 2 KB D-Cache, and core SRAM.
* **Bus & Memory Interfaces:**
  * ACE-Lite memory interface.
  * System Level Cache (SLC): 2048 KB.
  * 4-channel memory bus, each channel 256 bits wide (total 1024-bit memory bus width).
* **PVRIC Version:** Version 5 (FBCDC) with full rate 2-plane YUV and 10-bit sRGB/YUV support.
* **Configurable Parameters:**
  * Core Integrator ID.
  * SOCIF AXI Tag ID width.
  * Single or multiple clock domain configurations.
  * Single Core Support.
  * AXI memory features.

### 4.2 3D Workload Configuration
* **ISP Ratio:** 1 ISP per USC (8 ISPs total).
* **Tiles in Flight:** Up to 4 tiles per ISP.
* **MSAA Support:** Maximum 8× MSAA.
* **Tile IDs:** 12 reusable tile IDs per USC.
* **Depth Buffers:** 8 on-chip depth buffers per ISP for non-MSAA renders (4 for MSAA renders).
* **USC Controller Task Space:** 61.

### 4.3 USC Pipelines Performance & Internal Storage Sizing

#### Pipeline Throughput (per USC)
* **PAP Program Instances:** 128 instances/clock
* **Iteration Operations:** 32 operations/clock
* **RCP / RSQ Operations:** 32 operations/clock
* **Sample Operations:** 8 operations/clock
* **UVB Write Operations:** 4 operations/clock
* **Local Memory 32-bit Atomics:** 4 operations/clock
* **USC Allocatable Slots:** 48 maximum slots (up to 128 instances per slot)

#### Internal On-Chip Storage Specifications

| Storage Unit | Description / Function | Capacity (DWORDS / KB) | Organization / Banks | Max Allocation Limit | Granularity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Coefficient Store (CS)** | Stores coefficient registers | 5,856 DWORDS (22.875 KB) | 1 bank | Max 396 32-bit registers (132 varyings) / primitive | 12 DWORDS (4 varyings); Max 488 regions |
| **Local Memory Store (LMS)** | Stores local memory registers (LM) | 8,192 DWORDS (32 KB) | 16 banks | Max 8,192 32-bit registers / shader task | 16 DWORDS; Max 512 regions |
| **Shared Store (SHS)** | Stores shared registers (SH) | 5,120 DWORDS (20 KB) | 1 bank | Max 2,048 32-bit registers / shader task | 64 DWORDS; Max 80 regions |
| **Unified Store (US)** | Stores temporary & attribute registers | 131,072 DWORDS (512 KB) total | 128 instances × 1,024 DWORDS (4 KB each) | Shared between attributes and temps | 2 DWORDS; Max 512 regions |
| **Slot Registers Store (SRS)** | Stores slot registers | 192 DWORDS | 48 allocatable slots | 4 slot registers (1 DWORD each) / slot | Dedicated |
| **Internal Registers Store (IRS)** | Stores internal USC state | 12,288 DWORDS (48 KB) | USC instances | 2 internal registers / slot / instance | 2 DWORDS |
| **USC L2 Cache** | USC L2 cache storage | 8 KB | 64 bytes wide × 128 deep | – | – |
| **Partition Store Blend (PSB)** | Pixel blending processing | – | 16 instances / cycle | – | – |
| **Partition Store (PS)** | Stores pixel output registers | 32,768 DWORDS (128 KB) per USC in SPU | 8 banks (32 bytes wide × 1024 deep) | Max 8 pixel output registers / sample; Partition: 2,048 DWORDS (8 KB) | – |
| **USC Vertex Buffer (UVB)** | Stores geometry vertices | 32,768 DWORDS (128 KB) | – | 512 allocation regions for vertices | 64 DWORDS |
| **PDS Store** | Stores constant & temp registers for sequencer | 2,048 DWORDS (8 KB) | – | Max 192 32-bit constant/temps (up to 32 temps/instance) | 8 DWORDS (4 × 64-bit); Max 256 regions |

### 4.4 System Memory Hierarchy, Caches & MMU

#### Cache Configurations
* **System Level Cache (SLC):**
  * Size: **2048 KB**
  * Cache Line Size: **1024 bits (128 bytes)**
  * Banks: **8 cache banks**
* **Mixed Cache Unit (MCU) L1:**
  * Size: **24 KB**
  * Cache Line Size: **512 bits (64 bytes)**
  * Banks: **4 MCU banks** (96 cachelines per bank)
* **Texture Cache Unit (TCU):**
  * Size: **24 KB**
  * Cache Line Size: **512 bits (64 bytes)**
  * Banks: **4 TCU banks** (96 cachelines per bank)
* **Texture Processing Unit (TPU):**
  * Dedicated full-rate 2-plane YUV hardware support.

#### Memory Management Unit (MMU) Configuration
* **Page Catalogue Cache:** 256 entries
* **Page Directory Cache:** 1,024 entries
* **Page Table Cache:** 4,096 entries
* **Virtual Address Width:** 40 bits (up to 1 TB addressable space)

---

## 5. Performance Characteristics & Formulas

### 5.1 Peak Theoretical Performance (100% Efficiency)

| Feature / Metric | Theoretical Peak Throughput |
| :--- | :--- |
| **Floating Point Operations (F32)** | **2,048 operations / clock** |
| **Floating Point Operations (F16)** | **4,096 operations / clock** |
| **Integer Operations (INT)** | **512 operations / clock** |
| **Neural Network Compute (DOT8 NN)** | **8,192 operations / clock** |
| **Geometry Performance** | **2.0 triangles / clock** |
| **Texture Performance** | **64 texels / clock** (@ 32 BPP) |
| **Pixel Performance** | **64 pixels / clock** (@ 32 BPP) |

### 5.2 Performance Formulas & Derivations

1. **Floating Point Operations (F32):**
   $$\text{F32 Ops/Clock} = N_{\text{USC}} \times N_{\text{Parallel FP Instances/USC}} \times N_{\text{F32 Ops/Instance}}$$
   $$\text{F32 Ops/Clock} = 8 \times 128 \times 2 = 2048 \text{ ops/clock}$$

2. **Floating Point Operations (F16):**
   $$\text{F16 Ops/Clock} = 8 \times 128 \times 4 = 4096 \text{ ops/clock}$$

3. **Integer Operations (INT):**
   $$\text{INT Ops/Clock} = N_{\text{USC}} \times N_{\text{Parallel INT Instances/USC}} \times N_{\text{INT Ops/Instance}}$$
   $$\text{INT Ops/Clock} = 8 \times 32 \times 2 = 512 \text{ ops/clock}$$

4. **Neural Network Compute (DOT8 NN):**
   $$\text{DOT8 Ops/Clock} = N_{\text{USC}} \times N_{\text{Parallel DOT8 Instances/USC}} \times N_{\text{DOT8 Ops/Instance}}$$
   $$\text{DOT8 Ops/Clock} = 8 \times 128 \times 8 = 8192 \text{ ops/clock}$$

---

## 6. Top-Level Interface Characteristics

### 6.1 AXI Slave SoC Interface Specification (axi3)

| Feature Type | Feature Implementation Specification |
| :--- | :--- |
| **AXI Type** | AXI3 |
| **Role** | Slave |
| **Burst Attribute** | Bursts are not supported on the SOCIF |
| **Address Bus Width** | 32 bits |
| **Data Bus Width** | 32 bits |
| **Tag ID Width** | `AXIBUS_SOCIF_TAGIDS_WIDTH` |
| **Number of IDs** | $2^{\text{AXIBUS\_SOCIF\_TAGIDS\_WIDTH}}$ |
| **Max Outstanding Reads** | 24 |
| **Max Outstanding Writes** | 5 |
| **Write Interleaving** | Write Interleaving is not supported |
| **Sideband Signal** | N/A |
| **Unaligned Transfer** | Not supported |

---
*End of Specification — IMG D-Series DXTP-64-2048 Technical Specification (Revision 1.34)*
