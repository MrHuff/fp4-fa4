template <typename C>
struct globals_mxfp4_nb128_dvo128_qstage2_two_n64 {
    using base = globals_fp4pv_mxfp4_dv<C, true>;
    using q_tile = typename base::q_tile;
    using q_sc_tile = typename base::q_sc_tile;
    using k_tile = st_fp4e2m1_2<C::Nb, C::Dqk / 2>;
    using physical_k_tile = st_fp4e2m1_2<128, C::Dqk / 2>;
    using k_sc_tile = typename base::k_sc_tile;
    using p_fp4_tile = typename base::p_fp4_tile;
    using p_sc_tile = typename base::p_sc_tile;
    using v_fp4_tile = st_fp4e2m1_2<C::Dvo, C::Nb / 2>;
    using v_sc_tile = typename base::v_sc_tile;
    using physical_v_tile = st_fp4e2m1_2<C::Dvo, 64>;
    using physical_k_sc_tile = typename base::k_sc_tile;
    using physical_v_sc_tile = typename base::v_sc_tile;
    const uint8_t *q_fp4, *q_sc, *k_fp4, *k_sc;
    const float *q_sg, *k_sg;
    const uint8_t *v_fp4, *v_sc;
    bf16 *o; float *lse; unsigned long long *trace;
    int query_tiles, heads, seqlen;
    float softmax_scale_log2;
    __host__ __inline__ dim3 grid() const { return dim3(((query_tiles + 1) / 2) * heads); }
    __host__ __inline__ dim3 block() const { return dim3(C::NUM_THREADS); }
private:
    static constexpr int smem_size() {
        int bytes = 0;
        auto add = [&](int n) constexpr { bytes = align_up_constexpr(bytes, 1024) + n; };
        add(sizeof(q_tile)); add(sizeof(q_tile));
        add(sizeof(q_sc_tile)); add(sizeof(q_sc_tile));
        add(sizeof(physical_k_tile)); add(sizeof(physical_k_tile));
        add(sizeof(physical_k_sc_tile)); add(sizeof(physical_k_sc_tile));
        add(sizeof(p_fp4_tile)); add(sizeof(p_fp4_tile));
        add(sizeof(p_sc_tile)); add(sizeof(p_sc_tile));
        add(sizeof(physical_v_tile)); add(sizeof(physical_v_tile));
        add(sizeof(physical_v_sc_tile)); add(sizeof(physical_v_sc_tile));
        return bytes + 1024;
    }
public:
    __host__ __inline__ int dynamic_shared_memory() const { return smem_size(); }
};

template <typename C> __launch_bounds__(512, 1)
__global__ void kernel_mxfp4_nb128_dvo128_qstage2_two_n64(
    const __grid_constant__ globals_mxfp4_nb128_dvo128_qstage2_two_n64<C> g) {
    using G = globals_mxfp4_nb128_dvo128_qstage2_two_n64<C>;
    constexpr int SCORE_BASE=0, OQ0_BASE=128, OQ1_BASE=256;
    constexpr int Q_SCALE_BASE=384, K_SCALE_BASE=400, P_SCALE_BASE=416, V_SCALE_BASE=432;
    static_assert(C::NUM_THREADS == 512 && C::TOTAL_WGS == 4);
    static_assert(V_SCALE_BASE + 16 <= tensor_allocator<1, 1>::cols);
    tensor_allocator<1, 1> tm_alloc{};
    using score_t=tt<float,128,64>; using output_t=tt<float,128,128>;
    score_t tt_score0 = tm_alloc.template allocate<score_t>(0);
    score_t tt_score1 = tm_alloc.template allocate<score_t>(64);
    auto score_for_stage = [&](int stage) -> score_t & { return stage == 0 ? tt_score0 : tt_score1; };
    score_t &tt_score = tt_score0;
    output_t tt_output0 = tm_alloc.template allocate<output_t>(OQ0_BASE);
    output_t tt_output1 = tm_alloc.template allocate<output_t>(OQ1_BASE);
    using qk_scale_t = full_tt_fp8e4m3<16>;
    using pv_scale_t = full_tt_fp8e8m0<16>;
    qk_scale_t q_sc_tm = tm_alloc.template allocate<qk_scale_t>(Q_SCALE_BASE);
    qk_scale_t k_sc_tm = tm_alloc.template allocate<qk_scale_t>(K_SCALE_BASE);
    pv_scale_t p_sc_tm = tm_alloc.template allocate<pv_scale_t>(P_SCALE_BASE);
    pv_scale_t v_sc_tm = tm_alloc.template allocate<pv_scale_t>(V_SCALE_BASE);
    extern __shared__ int qstage2_smem[];
    tma_swizzle_allocator al((int *)&qstage2_smem[0]);
    typename G::q_tile &q_smem0 = al.allocate<typename G::q_tile>();
    typename G::q_tile &q_smem1 = al.allocate<typename G::q_tile>();
    typename G::q_sc_tile &q_sc_smem0 = al.allocate<typename G::q_sc_tile>();
    typename G::q_sc_tile &q_sc_smem1 = al.allocate<typename G::q_sc_tile>();
    using k_tile_t = typename G::k_tile;
    using physical_k_tile_t = typename G::physical_k_tile;
    using physical_v_tile_t = typename G::physical_v_tile;
    physical_k_tile_t &k_full0 = al.allocate<physical_k_tile_t>();
    physical_k_tile_t &k_full1 = al.allocate<physical_k_tile_t>();
    typename G::physical_k_sc_tile &k_sc_full0 = al.allocate<typename G::physical_k_sc_tile>();
    typename G::physical_k_sc_tile &k_sc_full1 = al.allocate<typename G::physical_k_sc_tile>();
    typename G::p_fp4_tile &p_smem0 = al.allocate<typename G::p_fp4_tile>();
    typename G::p_fp4_tile &p_smem1 = al.allocate<typename G::p_fp4_tile>();
    typename G::p_sc_tile &p_sc_smem0 = al.allocate<typename G::p_sc_tile>();
    typename G::p_sc_tile &p_sc_smem1 = al.allocate<typename G::p_sc_tile>();
    physical_v_tile_t &v_full0 = al.allocate<physical_v_tile_t>();
    physical_v_tile_t &v_full1 = al.allocate<physical_v_tile_t>();
    typename G::physical_v_sc_tile &v_sc_full0 = al.allocate<typename G::physical_v_sc_tile>();
    typename G::physical_v_sc_tile &v_sc_full1 = al.allocate<typename G::physical_v_sc_tile>();
    auto load_payload = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        constexpr int bytes = 4;
        static_assert(Tile::cols % bytes == 0);
        constexpr int stores = Tile::rows * (Tile::cols / bytes);
        const uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
        for (int i=warpgroup::laneid(); i<stores; i+=128) {
            const int row=i/(Tile::cols/bytes), col=(i%(Tile::cols/bytes))*bytes;
            const uint32_t value=*reinterpret_cast<const uint32_t *>(src+row*Tile::cols+col);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(Tile::idx(base,{row,col})),"r"(value));
        }
    };
    auto load_scale_record = [&]<typename Tile>(Tile &tile, const uint8_t *src) {
        const uint32_t base=__cvta_generic_to_shared(&tile.data[0]);
        for (int i=warpgroup::laneid();i<(int)sizeof(Tile)/4;i+=128) {
            const uint32_t value=*reinterpret_cast<const uint32_t *>(src+i*4);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(base+i*4),"r"(value));
        }
    };
    const int block_query=blockIdx.x/g.heads, h=blockIdx.x%g.heads;
    const float head_softmax_scale_log2 = g.softmax_scale_log2 * g.q_sg[h] * g.k_sg[h];
    const int query_tile[2]={2*block_query,2*block_query+1};
    const bool valid[2]={query_tile[0]<g.query_tiles,query_tile[1]<g.query_tiles};
    __shared__ semaphore kv_full[2], kv_empty[2];
    __shared__ semaphore qk_done[2][2];
    __shared__ semaphore score_released[2][2];
    __shared__ semaphore output_rescaled[2][2];
    __shared__ semaphore p_ready[2][2];
    __shared__ semaphore pv_src_done[2][2];
    __shared__ semaphore pv_done[2][2];
    __shared__ float stage_acc_scale[2][128];
    __shared__ int k_load_count;
    __shared__ int v_load_count;
    __shared__ int qk_issue_count[2];
    __shared__ int pv_issue_count[2];
    __shared__ int qk_half_count[2];
    __shared__ int pv_half_count[2];
    if(threadIdx.x==0){for(int slot=0;slot<2;++slot){init_semaphore(kv_full[slot],0,1);init_semaphore(kv_empty[slot],0,1);} for(int s=0;s<2;++s)for(int slot=0;slot<2;++slot){init_semaphore(qk_done[s][slot],0,1);init_semaphore(score_released[s][slot],0,1);init_semaphore(output_rescaled[s][slot],0,1);init_semaphore(p_ready[s][slot],0,1);init_semaphore(pv_src_done[s][slot],0,1);init_semaphore(pv_done[s][slot],0,1);}
        k_load_count=0; v_load_count=0; qk_issue_count[0]=0; qk_issue_count[1]=0; pv_issue_count[0]=0; pv_issue_count[1]=0; qk_half_count[0]=0; qk_half_count[1]=0; pv_half_count[0]=0; pv_half_count[1]=0;
        g.trace[blockIdx.x*32+0]=block_query; g.trace[blockIdx.x*32+1]=valid[0]; g.trace[blockIdx.x*32+2]=valid[1];}
    __syncthreads();
    for (int stage=0; stage<2; ++stage) if (valid[stage]) {
        auto load_q = [&](auto &qt, auto &qs) {
            load_payload(qt, g.q_fp4 + (query_tile[stage]*g.heads+h)*sizeof(typename G::q_tile));
            load_scale_record(qs, g.q_sc + (query_tile[stage]*g.heads+h)*sizeof(typename G::q_sc_tile));
        };
        if (warpgroup::groupid() == 3) { if (stage == 0) load_q(q_smem0, q_sc_smem0); else load_q(q_smem1, q_sc_smem1); }
    }
    __syncthreads();
    const int last_key=valid[1]?query_tile[1]:query_tile[0];
    float row_max_state[2] = {base_types::constants<float>::neg_infty(), base_types::constants<float>::neg_infty()};
    float row_sum_state[2] = {0.0f, 0.0f};
    auto load_v_once = [&](int key_tile, auto &v_slot, auto &v_sc_slot) {
        using VTile = typename G::v_fp4_tile;
        constexpr int bytes = 4;
        constexpr int stores = VTile::rows * (VTile::cols / bytes);
        const int packed_seqlen = g.seqlen / 2;
        const uint32_t base = __cvta_generic_to_shared(&v_slot.data[0]);
        for (int linear=warpgroup::laneid(); linear<stores; linear+=128) {
            const int row=linear/(VTile::cols/bytes), col=(linear%(VTile::cols/bytes))*bytes;
            const int global_row=h*C::Dvo+row;
            const uint32_t value=*reinterpret_cast<const uint32_t *>(g.v_fp4+global_row*packed_seqlen+key_tile*64+col);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(VTile::idx(base,{row,col})),"r"(value));
        }
        load_scale_record(v_sc_slot, g.v_sc+(h*g.query_tiles+key_tile)*sizeof(typename G::v_sc_tile));
    };
    auto issue_stage_qk = [&](auto &q_tile_ref, auto &q_sc_ref, auto &k_slot,
                              auto &k_sc_slot, int stage, int slot, int phase) {
        using T_AB = typename G::q_tile::T;
        using T_SAB = typename qk_scale_t::T;
        using T_D = typename score_t::T;
        constexpr uint32_t IDESC = detail::tcgen05::instruction_descriptor<T_D,T_AB,T_SAB,C::Mb,C::Nb,false,0>();
        st_descriptor<typename G::q_tile,0> q_desc(q_tile_ref);
        using KView = std::remove_reference_t<decltype(k_slot)>;
        st_descriptor<KView,0> k_desc(k_slot);
        const bool issue_warp = warpgroup::groupid()==0 && warpgroup::warpid()==0;
        const bool issue_lane = issue_warp && warp::elect_leader();
        for (int chunk=0; chunk<3; ++chunk) {
            if (issue_lane) {
                fp4pv_load_q_scale_chunk_ranklocal(q_sc_tm, q_sc_ref, chunk);
                fp4pv_load_k_scale_chunk_ranklocal(k_sc_tm, k_sc_slot, chunk);
            }
            if (issue_warp) { tensor_load_wait(); tensor_before_thread_sync(); }
            if (issue_lane) {
                if (chunk == 0) {
                detail::tcgen05::template st_st<T_AB,T_SAB,0,1,16>(
                    score_for_stage(stage).addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                    q_sc_tm.addr, k_sc_tm.addr, IDESC);
                } else {
                detail::tcgen05::template st_st<T_AB,T_SAB,1,1,16>(
                    score_for_stage(stage).addr, q_desc.chunk_descriptor(chunk), k_desc.chunk_descriptor(chunk),
                    q_sc_tm.addr, k_sc_tm.addr, IDESC);
                }
            }
        }
        if (issue_lane) {
            detail::tcgen05::commit<1>(qk_done[stage][slot]);
            atomicAdd(&qk_issue_count[stage], 1);
        }
        (void)phase;
    };
    auto finish_output = [&](int stage, output_t &tt_output) {
        if (warpgroup::groupid()!=stage+1) return;
        const int row = warpgroup::laneid();
        const bool finite = row_sum_state[stage] > 0.0f && isfinite(row_sum_state[stage]);
        const float scale = finite ? (1.0f/(36.0f * FP4PV_MXFP4_ONLINE_P_SCALE)) / row_sum_state[stage] : 0.0f;
        for (int col=0; col<128; col+=16) {
            float2 out[8];
            const uint32_t addr = tt_output.addr + ((warpgroup::warpid()*32)<<16) + col;
            asm volatile("{tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(out[0].x),"=f"(out[0].y),"=f"(out[1].x),"=f"(out[1].y),"=f"(out[2].x),"=f"(out[2].y),"=f"(out[3].x),"=f"(out[3].y),"=f"(out[4].x),"=f"(out[4].y),"=f"(out[5].x),"=f"(out[5].y),"=f"(out[6].x),"=f"(out[6].y),"=f"(out[7].x),"=f"(out[7].y) : "r"(addr));
            tensor_load_wait(); tensor_before_thread_sync();
            for (int i=0;i<8;++i) {
                const bf16_2 packed=__float22bfloat162_rn(__fmul2_rn(out[i],float2{scale,scale}));
                const int out_offset=(((query_tile[stage]*128+row)*g.heads+h)*128)+col+i*2;
                *reinterpret_cast<bf16_2 *>(&g.o[out_offset])=packed;
            }
        }
        const bool lse_valid=row_sum_state[stage]>0.0f && isfinite(row_sum_state[stage]);
        float lse_value=base_types::constants<float>::neg_infty();
        if (lse_valid) { float log2_sum; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(log2_sum) : "f"(row_sum_state[stage])); lse_value=(row_max_state[stage]*head_softmax_scale_log2+log2_sum)*0.693147180559945f; }
        g.lse[((h*g.seqlen) + query_tile[stage]*128 + row)] = lse_value;
    };
    for(int key_tile=0;key_tile<=last_key;++key_tile){
        auto &k_full = (key_tile & 1) ? k_full1 : k_full0;
        auto &k_sc_full = (key_tile & 1) ? k_sc_full1 : k_sc_full0;
        auto &v_full = (key_tile & 1) ? v_full1 : v_full0;
        auto &v_sc_full = (key_tile & 1) ? v_sc_full1 : v_sc_full0;
        auto &k_smem = reinterpret_cast<k_tile_t &>(k_full);
        auto &k_sc_smem = reinterpret_cast<typename G::k_sc_tile &>(k_sc_full);
        auto &v_smem = reinterpret_cast<typename G::v_fp4_tile &>(v_full);
        auto &v_sc_smem = reinterpret_cast<typename G::v_sc_tile &>(v_sc_full);
        const int slot=key_tile&1, phase=(key_tile>>1)&1;
        if (warpgroup::groupid() == 3) {
            if (key_tile >= 2 && warpgroup::warpid() == 0 && warp::elect_leader()) {
                wait(kv_empty[slot], phase ^ 1);
            }
            warpgroup::sync(4);
            load_payload(k_full, g.k_fp4 + (h*g.query_tiles+key_tile)*sizeof(physical_k_tile_t));
            load_scale_record(k_sc_full, g.k_sc + (h*g.query_tiles+key_tile)*sizeof(typename G::physical_k_sc_tile));
            load_v_once(key_tile, v_full, v_sc_full);
            warpgroup::sync(4);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(4);
            if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                arrive(kv_full[slot]);
            }
        }
        if (threadIdx.x == 0) atomicAdd(&k_load_count, 1);
        if (threadIdx.x == 0) atomicAdd(&v_load_count, 1);
        if (warpgroup::groupid()==0 && warpgroup::warpid()==0 && warp::elect_leader()) {
            wait(kv_full[slot], phase);
        }
        for (int key_half=0; key_half<2; ++key_half) {
            auto k_view = k_full.template subtile<64, C::Dqk / 2>(int2{key_half * 64, 0});
            auto v_view = v_full.template subtile<C::Dvo, C::Nb / 2>(int2{0, key_half * (C::Nb / 2)});
            for(int stage=0;stage<2;++stage) if(valid[stage]&&key_tile<=query_tile[stage]){
            auto &p_smem = stage == 0 ? p_smem0 : p_smem1;
            auto &p_sc_smem = stage == 0 ? p_sc_smem0 : p_sc_smem1;
            if(stage==0) issue_stage_qk(q_smem0,q_sc_smem0,k_view,k_sc_smem,stage,slot,phase);
            else issue_stage_qk(q_smem1,q_sc_smem1,k_view,k_sc_smem,stage,slot,phase);
            atomicAdd(&qk_half_count[stage], 1);
            if (warpgroup::groupid()==stage+1) {
                wait(qk_done[stage][slot],phase);
                tensor_after_thread_sync();
                const uint32_t score_tt_base = tt_score.addr + ((warpgroup::warpid()*32)<<16);
                float block_max[4]; float tile_max=base_types::constants<float>::neg_infty();
                const bool diagonal = key_tile == query_tile[stage];
                for(int qid=0;qid<4;++qid){float2 s[16];fp4pv_load_score_quarter_tmem(s,score_tt_base,qid);tensor_load_wait();float m=base_types::constants<float>::neg_infty();for(int i=0;i<16;++i){int kc=key_tile*128+qid*32+i*2;if(diagonal&&kc>query_tile[stage]*128+warpgroup::laneid())s[i].x=base_types::constants<float>::neg_infty();if(diagonal&&kc+1>query_tile[stage]*128+warpgroup::laneid())s[i].y=base_types::constants<float>::neg_infty();m=fmaxf(m,fmaxf(s[i].x,s[i].y));}block_max[qid]=m;tile_max=fmaxf(tile_max,m);}
                const float old_max=row_max_state[stage]; row_max_state[stage]=fmaxf(old_max,tile_max); const float acc_scale=key_tile?softmax_rescale_factor_strict(row_max_state[stage],old_max,head_softmax_scale_log2):1.0f; stage_acc_scale[stage][warpgroup::laneid()]=acc_scale;
                uint32_t packed_scales=0; float tile_sum=0.0f;
                for(int qid=0;qid<4;++qid){uint8_t pe=0;if(isfinite(block_max[qid])&&isfinite(row_max_state[stage])) pe=fp4pv_log2_to_e8m0_floor_finite(FP4PV_MXFP4_ONLINE_LOG2_P_SCALE+(block_max[qid]-row_max_state[stage])*head_softmax_scale_log2);packed_scales|=(uint32_t)pe<<(qid*8);float rc=(pe==0)?0.0f:fp4pv_exp2f_e8m0(pe)*FP4PV_MXFP4_ONLINE_RCP_6P_SCALE;float bias=-row_max_state[stage]*head_softmax_scale_log2+((pe==0)?0.0f:(FP4PV_MXFP4_ONLINE_LOG2_6P_SCALE-(float)((int)pe-127)));float2 sscale={head_softmax_scale_log2,head_softmax_scale_log2},bias2={bias,bias};float2 s[16];fp4pv_load_score_quarter_tmem(s,score_tt_base,qid);tensor_load_wait();float2 sum={0,0};for(int i=0;i<16;++i){int kc=key_tile*128+qid*32+i*2;if(diagonal&&kc>query_tile[stage]*128+warpgroup::laneid())s[i].x=base_types::constants<float>::neg_infty();if(diagonal&&kc+1>query_tile[stage]*128+warpgroup::laneid())s[i].y=base_types::constants<float>::neg_infty();s[i]=__ffma2_rn(s[i],sscale,bias2);s[i].x=fp4pv_exp2_approx(s[i].x);s[i].y=fp4pv_exp2_approx(s[i].y);sum=__fadd2_rn(sum,s[i]);}tile_sum+=(sum.x+sum.y)*rc;uint32_t words[4];for(int p=0;p<4;++p)words[p]=(pe==0)?0u:fp4pv_cvt_fp32_to_fp4_8x_prescaled_rte(&s[p*4]);fp4pv_store_quantized_scores_group_mxfp4(p_smem,words,warpgroup::laneid(),qid);}
                fp4pv_store_mxfp4_scale_word_swizzled(p_sc_smem,warpgroup::laneid(),packed_scales); row_sum_state[stage]=row_sum_state[stage]*acc_scale+tile_sum;
                if (stage == 0) warpgroup::sync(6); else warpgroup::sync(7);
                fp4pv_publish_shared_backing_proxy_only();
                if (stage == 0) warpgroup::sync(6); else warpgroup::sync(7);
                if (warpgroup::warpid() == 0 && warp::elect_leader()) {
                    arrive(score_released[stage][slot]);
                    arrive(p_ready[stage][slot]);
                }
            }
            if (warpgroup::groupid()==stage+1) {
                if (key_tile > 0) {
                    rescale_tt_output_inplace_direct_streaming<C>(stage ? tt_output1 : tt_output0, stage_acc_scale[stage][warpgroup::laneid()], true);
                    tensor_load_wait();
                }
                if (stage == 0) warpgroup::sync(6); else warpgroup::sync(7);
                if (warpgroup::warpid() == 0 && warp::elect_leader()) arrive(output_rescaled[stage][slot]);
            }
            if (warpgroup::groupid()==0) {
                const bool issue_warp = warpgroup::warpid()==0;
                const bool issue_lane = issue_warp && warp::elect_leader();
                if (issue_lane) {
                    wait(score_released[stage][slot], phase);
                    wait(p_ready[stage][slot], phase);
                    wait(output_rescaled[stage][slot], phase);
                    fp4pv_load_mxnv_scale_async_ranklocal(p_sc_tm,p_sc_smem);
                    fp4pv_load_mxnv_scale_async_ranklocal(v_sc_tm,v_sc_smem);
                }
                if (issue_warp) { tensor_load_wait(); tensor_before_thread_sync(); }
                if (issue_lane) {
                    if (key_tile == 0 && key_half == 0) {
                        fp4pv_mm_p_stage_ABt<C>(stage ? tt_output1 : tt_output0,
                            p_smem, v_view, p_sc_tm, v_sc_tm, pv_src_done[stage][slot]);
                    } else {
                        fp4pv_mm_p_stage_ABt<C>(stage ? tt_output1 : tt_output0,
                            p_smem, v_view, p_sc_tm, v_sc_tm, pv_src_done[stage][slot]);
                    }
                    tensor_commit<1>(pv_done[stage][slot]);
                    atomicAdd(&pv_issue_count[stage], 1);
                    atomicAdd(&pv_half_count[stage], 1);
                }
            }
            if (warpgroup::groupid()==stage+1) {
                wait(pv_done[stage][slot],phase);
                tensor_after_thread_sync();
            }
            }
        }
        if (warpgroup::groupid()==0 && warpgroup::warpid()==0 && warp::elect_leader()) {
            const int last_consumer_stage = valid[1] ? 1 : 0;
            wait(pv_src_done[last_consumer_stage][slot], phase);
            arrive(kv_empty[slot]);
        }
    }
    if (valid[0]) finish_output(0,tt_output0);
    if (valid[1]) finish_output(1,tt_output1);
    __syncthreads();
    if(threadIdx.x==0){g.trace[blockIdx.x*32+3]=last_key; g.trace[blockIdx.x*32+6]=k_load_count; g.trace[blockIdx.x*32+7]=qk_issue_count[0]; g.trace[blockIdx.x*32+8]=qk_issue_count[1]; g.trace[blockIdx.x*32+9]=pv_issue_count[0]; g.trace[blockIdx.x*32+10]=pv_issue_count[1]; g.trace[blockIdx.x*32+11]=v_load_count; g.trace[blockIdx.x*32+12]=qk_half_count[0]; g.trace[blockIdx.x*32+13]=qk_half_count[1]; g.trace[blockIdx.x*32+14]=pv_half_count[0]; g.trace[blockIdx.x*32+15]=pv_half_count[1];}
}
