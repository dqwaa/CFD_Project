# ============================================================================
# cylinder-flow-newton.jl — 二维不可压圆柱绕流 (Cylinder Flow)
# 基于 ApproxOperator.jl + 残差修正 Newton-Raphson 迭代 + 后向欧拉时间推进
# ============================================================================

# ========================== Section 1: Dependencies ==========================

using ApproxOperator
using ApproxOperator.GmshImport: getPhysicalGroups, get𝑿ᵢ, getElements
using WriteVTK
using SparseArrays, LinearAlgebra
using Printf
using Statistics

import ApproxOperator.Stokes: ∫∫μ∇u∇vdxdy,  ∫∫ρvdxdy, ∫∫ρ∇uvudxdy, update_velocity
import ApproxOperator.Elasticity: ∫∫p∇udxdy, ∫vᵢgᵢds
import Gmsh: gmsh

# ====================== Section 2: 物理 & 数值参数 ============================

# --- 物理参数 ---
const μ  = 0.01        # 动力粘性系数 
const ρ  = 1.0          # 密度
const U₀ = 1.0          # 来流特征速度
const D  = 1.0          # 特征长度
const Re = ρ * U₀ * D / μ
@printf("Reynolds number: Re = %.2f\n", Re)

# --- 网格配置文件路径 ---
const mesh_file_u   = "cfd/msh/tri/cylinder_tri_0.4-1.2refine.msh"  
const mesh_file_p   = "cfd/msh/tri/cylinder_tri_0.4-1.2.msh"

const mesh_type     = "tri"       
const intOrder      = 2            # 高斯积分阶数

const type_p = :(ReproducingKernel{:Linear2D,:□,:CubicSpline})
const TypeP  = eval(type_p)   

# --- VTK 输出配置 ---
const outdir    = "cfd/VTK/cylinder"
const case_name = "cylinder_newton"

# --- 边界条件物理组名映射 ---
const inlet    = "Γ₁"          
const outlet   = "Γ₂"          
const top      = "Γ₃"
const bottom   = "Γ₄"
const cylinder = "Γ₅"

# --- 时间推进与非线性求解参数 ---
const Δt           = 0.002      
const nsteps       = 15000     
const vtk_interval = 50          # VTK 输出间隔步数       

const maxNewton    = 20       
const newtonTol    = 1e-5     
const T_ramp       = 1.0       # 三次光滑阶跃启动时间窗口
const H_half       = 5.0       # 通道半高 (入口 y ∈ [-5, 5])

# ======================== Section 3: 网格加载与预处理 ==========================

# ---- 自适应支撑域工具函数 ----
function set_adaptive_support!(nodes_p, sp::ApproxOperator.RegularGrid; k_nearest=4, α=2.5)
    n = length(nodes_p)
    h_local    = zeros(n)
    n_support  = zeros(Int, n)    # 每个节点支撑域半径内覆盖的邻点数
    for (i, node) in enumerate(nodes_p)
        x, y, z = node.x, node.y, node.z
        candidates = sp(x, y, z)
        dists = Float64[]
        for j in candidates
            j == i && continue
            nbr = nodes_p[j]
            push!(dists, sqrt((x - nbr.x)^2 + (y - nbr.y)^2 + (z - nbr.z)^2))
        end
        sort!(dists)
        k = min(k_nearest, length(dists))
        if k == 0
            h_local[i] = 0.01; n_support[i] = 0; continue
        end
        h_local[i] = dists[k]
        r_support = α * h_local[i]
        n_support[i] = count(d -> d <= r_support, dists)
    end
    push!(nodes_p, :s₁     => α .* h_local,
                   :s₂     => α .* h_local,
                   :s₃     => α .* h_local,
                   ) 

    total_covered_nodes = n_support .+ 1 
    
    @info "=================== RKPM 支撑域统计报告 ==================="
    @info "  • 压力场总节点数 (Total Nodes)    : $n 个"
    @info "  • 局部节点间距 (Local Spacing h) : min=$(minimum(h_local)), max=$(maximum(h_local)), avg=$(mean(h_local))"
    @info "  • 每个支撑域覆盖的节点数 (Coverage) : min=$(minimum(total_covered_nodes)), max=$(maximum(total_covered_nodes)), avg=$(round(mean(total_covered_nodes), digits=1)) 个"
    @info "=========================================================="
end   


gmsh.initialize()

# ---- 3.1 压力网格 (RKPM, 粗网格) ----
@info "Loading pressure mesh..."
gmsh.open(mesh_file_p)
nodes_p    = get𝑿ᵢ()
entities = getPhysicalGroups()   
xᵖ, yᵖ, zᵖ = nodes_p.x, nodes_p.y, nodes_p.z
nᵖ = length(nodes_p)

sp = RegularGrid(xᵖ, yᵖ, zᵖ; n=8, γ=4)
set_adaptive_support!(nodes_p, sp; k_nearest=4, α=2.5)

# ---- 3.2 速度网格 (FEM, 细网格) ----
@info "Loading velocity mesh..."
gmsh.clear()  
gmsh.open(mesh_file_u)
entities = getPhysicalGroups()
nodes    = get𝑿ᵢ()
nᵘ       = length(nodes)

@info "Available physical groups:" keys(entities)

# ---- 3.3 提取单元 ----
@info "Extracting elements..."
elements_u  = getElements(nodes,    entities["Ω"],   intOrder)
elements_p  = getElements(nodes_p,  entities["Ω"],  TypeP, intOrder, sp)

elements_inlet  = getElements(nodes, entities[inlet],  intOrder)
elements_outlet = getElements(nodes, entities[outlet], intOrder)
elements_top    = getElements(nodes, entities[top],    intOrder)
elements_bottom = getElements(nodes, entities[bottom], intOrder)
elements_cylinder = getElements(nodes, entities[cylinder], intOrder; normal=true)

elements_vtk = getElements(nodes, entities["Ω"], intOrder)

# ---- 3.4 积分点参数初始化 ----
prescribe!(elements_u, :μ => μ, :ρ => ρ, :Δt => Δt)
prescribe!(elements_u, :u₁   => 0.0, :u₂   => 0.0,
                       :∂u₁∂x => 0.0, :∂u₁∂y => 0.0,
                       :∂u₂∂x => 0.0, :∂u₂∂y => 0.0)

# ---- 3.5 边界条件 (罚函数法) ----
const α_pen = 1e8


prescribe!(elements_inlet, :g₁ => U₀, :g₂ => 0.0, :α   => α_pen,
                           :n₁₁ => 1.0, :n₂₂ => 1.0, :n₁₂ => 0.0)
prescribe!(elements_outlet, :g₁ => 0.0, :g₂ => 0.0, :α   => 0.0,
                            :n₁₁ => 0.0, :n₂₂ => 0.0, :n₁₂ => 0.0)         
prescribe!(elements_top,    :g₁ => 0.0, :g₂ => 0.0, :α => α_pen,
                            :n₁₁ => 0.0, :n₂₂ => 1.0, :n₁₂ => 0.0)
prescribe!(elements_bottom, :g₁ => 0.0, :g₂ => 0.0, :α => α_pen,
                            :n₁₁ => 0.0, :n₂₂ => 1.0, :n₁₂ => 0.0)
prescribe!(elements_cylinder, :g₁ => 0.0, :g₂ => 0.0, :α => α_pen,
                              :n₁₁ => 1.0, :n₂₂ => 1.0, :n₁₂ => 0.0)


# ---- 3.6 计算形函数 ----
@info "Computing shape functions..."
set∇𝝭!(elements_u)    
set𝝭!(elements_p)     

all_boundary_elements = [
    elements_inlet, 
    elements_outlet, 
    elements_top,     
    elements_bottom,  
    elements_cylinder     
]

for elms in all_boundary_elements
    if !isempty(elms)
        push!(elms, :𝝭)
        for elm in elms
            for ξ in elm.𝓖
                set𝝭!(elm, ξ)
            end
        end
    end
end
elements_walls = reduce(∪, [elements_top, elements_bottom, elements_cylinder])

# ==================== Section 4: 全局矩阵 / 向量初始化 =========================

Kuu      = spzeros(2*nᵘ, 2*nᵘ)   # 全 Jacobian 中的速度块
Kuu_visc = spzeros(2*nᵘ, 2*nᵘ)   # 纯粘性刚度常数阵
Kup      = spzeros(nᵖ, 2*nᵘ)     # 散度/梯度耦合块
Kpp      = spzeros(nᵖ, nᵖ)       # 压力块

rhs_u    = zeros(2*nᵘ)         # 速度残差向量
rhs_p    = zeros(nᵖ)           # 压力(连续性)残差向量
tmp_vec  = zeros(2*nᵘ)         # 乘法辅助向量
f_g      = zeros(2*nᵘ)         # 非线性对流力向量

# 节点解向量
d₁      = zeros(nᵘ) 
d₂      = zeros(nᵘ) 
d₁_old  = zeros(nᵘ) 
d₂_old  = zeros(nᵘ) 
d₁_old2 = zeros(nᵘ)      # 两步前的解，用于外推初猜
d₂_old2 = zeros(nᵘ) 
p_vec   = zeros(nᵖ) 
push!(nodes,   :d₁ => d₁, :d₂ => d₂, :d₁_old => d₁_old, :d₂_old => d₂_old)
push!(nodes_p, :p => p_vec)

# ================== Section 5: 算子预定义 & 基础矩阵预计算 =======================

# 预计算罚项
@info "Precomputing penalty matrix and force..."
K_pen = spzeros(2*nᵘ, 2*nᵘ)
f_pen = spzeros(2*nᵘ)
bc_op = ∫vᵢgᵢds => (elements_inlet ∪ elements_walls)
bc_op(K_pen, f_pen)

# 算子配对
op_visc_mat = ∫∫μ∇u∇vdxdy => elements_u    
op_mass_t   = ∫∫ρvdxdy   => elements_u    
op_conv_mat = ∫∫ρ∇uvudxdy => elements_u    
op_pres_mat = ∫∫p∇udxdy   => (elements_p, elements_u)  

# 预计算质量矩阵 M_t
@info "Precomputing mass matrix M_t..."
M_t = spzeros(2*nᵘ, 2*nᵘ)
op_mass_t(M_t)

# 辅助函数：利用现有算子准确组装对流非线性力项 f^g(u^m)
function compute_convection_force!(f_g, op_conv_mat, u_m_vec)
    K_tmp = spzeros(length(f_g), length(f_g))
    op_conv_mat(K_tmp) 
    mul!(f_g, K_tmp, u_m_vec)
end

# ==================== Section 6: Newton-Raphson 求解器 =======================

function newton_step!(d₁, d₂, p_vec, d₁_old, d₂_old;
                       Kuu, Kuu_visc, Kup, Kpp, tmp_vec, rhs_u, rhs_p,
                       K_pen, f_pen, M_t, f_g, elements_u, op_conv_mat, op_pres_mat,
                       nᵘ, Δt, tol, maxiter)
    converged = false
    rel_err   = Inf
    iters     = 0

    u_n_vec = zeros(Float64, 2*nᵘ)
    u_n_vec[1:2:end] .= d₁_old
    u_n_vec[2:2:end] .= d₂_old
    
    u_m_vec = zeros(Float64, 2*nᵘ)

    # 2. 几何/线性矩阵提前组装 
    fill!(Kuu_visc, 0.0)
    op_visc_mat(Kuu_visc)
    
    fill!(Kup, 0.0)
    op_pres_mat(Kup)
    Kup_T = copy(Kup') # 提前计算好转置，避免循环中重复转置

    for m in 1:maxiter
        iters = m
        u_m_vec[1:2:end] .= d₁
        u_m_vec[2:2:end] .= d₂

        # =================== 组装 Jacobian ============
        fill!(Kuu, 0.0); fill!(Kpp, 0.0)

        Kuu .+= Kuu_visc          
        Kuu .+= M_t .* (1.0 / Δt) # 避免 ./ Δt 产生临时数组
        op_conv_mat(Kuu)          # 确保这里的对流矩阵是基于最新的 elements_u 线性化计算的
        Kuu .+= K_pen             

        # =================== 计算残差  =======================
        rhs_u .= f_pen

        @. tmp_vec = (u_m_vec - u_n_vec) / Δt
        mul!(rhs_u, M_t, tmp_vec, -1.0, 1.0)

        compute_convection_force!(f_g, op_conv_mat, u_m_vec)
        rhs_u .-= f_g

        mul!(tmp_vec, Kuu_visc, u_m_vec)
        rhs_u .-= tmp_vec

        mul!(tmp_vec, K_pen, u_m_vec)
        rhs_u .-= tmp_vec

        mul!(tmp_vec, Kup_T, p_vec)
        rhs_u .-= tmp_vec

        # 压力连续性残差
        fill!(rhs_p, 0.0)
        mul!(rhs_p, Kup, u_m_vec)
        rhs_p .*= -1.0
        
        # =================== 施加压力参考点 Dirichlet 边界条件 =====================
        # 【修正】在计算完物理残差后，再修改矩阵和残差，真正固定 p_1 = 0.0
        Kup_bc = copy(Kup)        # 避免修改原始物理矩阵
        Kup_bc[1, :] .= 0.0
        
        Kup_T_bc = copy(Kup_T)    
        Kup_T_bc[:, 1] .= 0.0     # 保持系统对称/正确解耦

        Kpp[1, 1] = 1.0           
        rhs_p[1]  = 0.0 - p_vec[1] # 这样解出的 Δp_1 会把 p_vec[1] 抵消到 0

        # =================== 求解 Newton 增量 ===================================
        # 注意：生产环境中这里应预先分配大矩阵 K，或者使用迭代求解器
        K = [Kuu      Kup_T_bc;
             Kup_bc   Kpp     ]
        RHS = [rhs_u; rhs_p]
        dx = K \ RHS

        Δu_vec = dx[1:2*nᵘ]
        Δp_vec = dx[2*nᵘ+1:end]

        # =================== Newton 更新 ========================================
        d₁ .+= Δu_vec[1:2:end]
        d₂ .+= Δu_vec[2:2:end]
        p_vec .+= Δp_vec
        
        # 使用 copy 防止引用污染历史记录
        push!(nodes,   :d₁ => copy(d₁), :d₂ => copy(d₂))
        push!(nodes_p, :p => copy(p_vec))

        for elm in elements_u
            update_velocity(elm)
        end

        # ---- 收敛判据 ----
        norm_du = norm(Δu_vec)
        norm_u  = norm(u_m_vec) + 1e-16
        rel_err = norm_du / norm_u
        @printf("  Newton iter %2d: |Δu|/|u| = %.3e, |r_u| = %.3e\n", m, rel_err, norm(rhs_u))

        if rel_err < tol
            converged = true
            break
        end
    end
    return converged, iters, rel_err
end

# ====================== Section 7: 时间推进主循环 ==============================

function solve_unsteady_navier_stokes!(d₁, d₂, p_vec, d₁_old, d₂_old, d₁_old2, d₂_old2,
                                       nodes, nodes_p, elements_u, elements_inlet,
                                       elements_vtk, sp, TypeP, outdir, case_name, nᵘ;
                                       Δt=Δt, nsteps=nsteps, vtk_interval=vtk_interval,
                                       T_ramp=T_ramp, H_half=H_half, U₀=U₀, α_pen=α_pen,
                                       newtonTol=newtonTol, maxNewton=maxNewton,
                                       Kuu=Kuu, Kuu_visc=Kuu_visc, Kup=Kup, Kpp=Kpp,
                                       tmp_vec=tmp_vec, rhs_u=rhs_u, rhs_p=rhs_p,
                                       K_pen=K_pen, f_pen=f_pen, M_t=M_t,
                                       f_g=f_g, bc_op=bc_op,
                                       op_conv_mat=op_conv_mat, op_pres_mat=op_pres_mat)

    mkpath(outdir)

    for step in 1:nsteps
        t = step * Δt
        @printf("\n--- Step %3d / %d (t = %.3f) ---\n", step, nsteps, t)

        # ---- 1. 设置入口边界条件（时空耦合抛物剖面） ----
        if t < T_ramp
            τ = t / T_ramp
            ramp_factor = τ^2 * (3.0 - 2.0τ)
        else
            ramp_factor = 1.0
        end

        prescribe!(elements_inlet, :g₁ => 0.0, :g₂ => 0.0, :α   => α_pen,
                                   :n₁₁ => 1.0, :n₂₂ => 1.0, :n₁₂ => 0.0)

        for elm in elements_inlet
            for ξ in elm.𝓖
                y = ξ.y
                spatial_u = U₀#开阔空间的流入速度
                # spatial_u = U₀ * (1.0 - (y / H_half)^2)狭窄空间，管道的流入速度
                ξ.g₁ = spatial_u * ramp_factor
            end
        end

        # 重新组装罚函数矩阵/力向量（入口边界条件随时间变化）
        fill!(K_pen, 0.0)
        fill!(f_pen, 0.0)
        bc_op(K_pen, f_pen)

        # ---- 2. Newton 初猜 (Predictor) ----
        if step == 1
            @. d₁ = d₁_old
            @. d₂ = d₂_old
        else
            @. d₁ = 2.0 * d₁_old - d₁_old2
            @. d₂ = 2.0 * d₂_old - d₂_old2
        end

        for elm in elements_u
            update_velocity(elm)
        end

        # ---- 3. Newton-Raphson 非线性求解 (Corrector) ----
        converged, iters, rel_err = newton_step!(
            d₁, d₂, p_vec, d₁_old, d₂_old;
            Kuu=Kuu, Kuu_visc=Kuu_visc, Kup=Kup, Kpp=Kpp,
            tmp_vec=tmp_vec, rhs_u=rhs_u, rhs_p=rhs_p,
            K_pen=K_pen, f_pen=f_pen, M_t=M_t,
            f_g=f_g, elements_u=elements_u,
            op_conv_mat=op_conv_mat, op_pres_mat=op_pres_mat,
            nᵘ=nᵘ,
            Δt=Δt,
            tol=newtonTol, maxiter=maxNewton
        )

        if !converged
            @warn "Newton did NOT converge in step $step (final rel_err = $(@sprintf("%.3e", rel_err)))"
            break
        else
            @printf("  Newton converged in %d iters, rel_err = %.3e\n", iters, rel_err)
        end

        # ---- 4. 时间步状态推进 ----
        @. d₁_old2 = d₁_old
        @. d₂_old2 = d₂_old
        @. d₁_old  = d₁
        @. d₂_old  = d₂
        push!(nodes,   :d₁_old => d₁_old, :d₂_old => d₂_old)

        # ---- 5. VTK 输出 ----
        if step % vtk_interval == 0 || step == nsteps
            @info "Writing VTK for step $step..."

            pressure = zeros(nᵘ)
            u₁_vtk = zeros(nᵘ)
            u₂_vtk = zeros(nᵘ)
            u₃_vtk = zeros(nᵘ)
            𝗠 = zeros(10)

            for (i, node) in enumerate(nodes)
                x, y, z = node.x, node.y, node.z
                indices = sp(x, y, z)
                ni = length(indices)
                pts = [nodes_p[j] for j in indices]

                data = Dict(
                    :x=>(2,[x]), :y=>(2,[y]), :z=>(2,[z]),
                    :𝝭=>(4,zeros(ni)), :𝗠=>(0,𝗠)
                )
                ξ = 𝑿ₛ((𝑔=1,𝐺=1,𝐶=1,𝑠=0), data)
                a_p = TypeP(pts, [ξ])
                set𝝭!(a_p)

                p_val = 0.0
                Np = ξ[:𝝭]
                for (k, xₖ) in enumerate(pts)
                    p_val += Np[k] * xₖ.p
                end
                pressure[i] = p_val
                u₁_vtk[i] = node.d₁
                u₂_vtk[i] = node.d₂
            end

            points = zeros(3, nᵘ)
            for node in nodes
                I = node.𝐼
                points[1, I] = node.x
                points[2, I] = node.y
                points[3, I] = node.z
            end

            cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, [xᵢ.𝐼 for xᵢ in elm.𝓒])
                     for elm in elements_vtk]

            filename = joinpath(outdir, "$(case_name)_step$(step).vtu")
            vtk_grid(filename, points, cells) do vtk
                vtk["u"] = (u₁_vtk, u₂_vtk, u₃_vtk)
                vtk["p"] = pressure
            end
        end
    end

    return nodes
end

# ====================== Section 7b: 启动计算 ===================================

solve_unsteady_navier_stokes!(d₁, d₂, p_vec, d₁_old, d₂_old, d₁_old2, d₂_old2,
                               nodes, nodes_p, elements_u, elements_inlet,
                               elements_vtk, sp, TypeP, outdir, case_name, nᵘ)

# ========================= Section 8: PVD 集合文件 ==========================

pvd_path = joinpath(outdir, "$(case_name).pvd")

open(pvd_path,"w") do io

    write(io,"<?xml version=\"1.0\"?>\n")

    write(io,"<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n")

    write(io,"<Collection>\n")

    for step in 1:nsteps

        if step % vtk_interval == 0 || step==nsteps


            fname = "$(case_name)_step$(step).vtu"

            write(
                io,
                "  <DataSet timestep=\"$(step*Δt)\" file=\"$(fname)\"/>\n"
            )

        end

    end

    write(io,"</Collection>\n")

    write(io,"</VTKFile>\n")

end

@info "Open $(pvd_path) in ParaView"
