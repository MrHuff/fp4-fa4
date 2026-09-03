/*
 * Copyright (C) 2025 Roberto L. Castro (Roberto.LopezCastro@ist.ac.at). All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <common.h>

namespace QUTLASS {

// Note: We use void* for pointers to avoid including cutlass headers here unnecessarily if not needed
// But implementation will cast them.
// We also need size arguments since we don't have Tensor::size()

void fusedQuantizeMxQuest_host(void* D_ptr,
                               void* D_sf_ptr,
                               void const* A_ptr,
                               void const* B_ptr);

void fusedQuantizeMxQuestWithMask_host(void* D_ptr,
                                       void* D_sf_ptr,
                                       void* D_mask_ptr,
                                       void const* A_ptr,
                                       void const* B_ptr);

void fusedQuantizeMxAbsMax_host(void* D_ptr,
                                void* D_sf_ptr,
                                void const* A_ptr,
                                void const* B_ptr);

void fusedQuantizeMxQuestHad64_host(void* D_ptr,
                                    void* D_sf_ptr,
                                    void const* A_ptr,
                                    void const* B_ptr);

void fusedQuantizeMxAbsMaxHad64_host(void* D_ptr,
                                     void* D_sf_ptr,
                                     void const* A_ptr,
                                     void const* B_ptr);

void fusedQuantizeMxQuestHad128_host(void* D_ptr,
                                     void* D_sf_ptr,
                                     void const* A_ptr,
                                     void const* B_ptr);

void fusedQuantizeMxAbsMaxHad128_host(void* D_ptr,
                                      void* D_sf_ptr,
                                      void const* A_ptr,
                                      void const* B_ptr);

void fusedQuantizeNvQuest_host(void* D_ptr,
                               void* D_sf_ptr,
                               void const* A_ptr,
                               void const* B_ptr,
                               void const* global_scale_ptr);

void fusedQuantizeNvQuestHad32_host(void* D_ptr,
                                    void* D_sf_ptr,
                                    void const* A_ptr,
                                    void const* B_ptr,
                                    void const* global_scale_ptr);

void fusedQuantizeNvQuestHad64_host(void* D_ptr,
                                    void* D_sf_ptr,
                                    void const* A_ptr,
                                    void const* B_ptr,
                                    void const* global_scale_ptr);

void fusedQuantizeNvQuestHad128_host(void* D_ptr,
                                     void* D_sf_ptr,
                                     void const* A_ptr,
                                     void const* B_ptr,
                                     void const* global_scale_ptr);

void fusedQuantizeNvAbsMax_host(void* D_ptr,
                                void* D_sf_ptr,
                                void const* A_ptr,
                                void const* B_ptr,
                                void const* global_scale_ptr);

void fusedQuantizeNvAbsMaxHad32_host(void* D_ptr,
                                     void* D_sf_ptr,
                                     void const* A_ptr,
                                     void const* B_ptr,
                                     void const* global_scale_ptr);

void fusedQuantizeNvAbsMaxHad64_host(void* D_ptr,
                                     void* D_sf_ptr,
                                     void const* A_ptr,
                                     void const* B_ptr,
                                     void const* global_scale_ptr);

void fusedQuantizeNvAbsMaxHad128_host(void* D_ptr,
                                      void* D_sf_ptr,
                                      void const* A_ptr,
                                      void const* B_ptr,
                                      void const* global_scale_ptr);

void fusedQuantizeMxAbsMax_host_sm100(void* D_ptr,
                                      void* D_sf_ptr,
                                      void const* A_ptr,
                                      void const* B_ptr,
                                      void const* global_scale_ptr);

void fusedQuantizeNvAbsMax_host_sm100(
                                      void* D_ptr,
                                      void* D_sf_ptr,
                                      void const* A_ptr,
                                      void const* B_ptr,
                                      void const* global_scale_ptr,
                                      int64_t A_numel,
                                      int64_t B_size1,
                                      cudaStream_t stream = 0);

}  // namespace QUTLASS
