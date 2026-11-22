# VertiRTA — 2-Page Cold-Derivation Whiteboard Reference

> **Whiteboard Lock:** Every equation below fits on 2 printed pages and is derivable from first principles without notes during a 45-minute technical interview. Nothing enters `libvertirta` if it violates this rule.

---

## Pillar 1: Coaxial X8 Allocation, 1-RC Sag, Discrete INDI + WLS & Option B1b Flat Spin

- **Coaxial Thrust & WLS 1-Motor Offloading (`VertiRTA-X8-240`, `m = 240 kg`, `W = 2353.6 N`, `T/W = 2.5`):**
  - Nominal max thrust `T_max = 5884.0 N` (`eta_coax = 0.85`): upper rotor `795.4 N`, lower rotor `676.1 N` (upper loss unblanks lower by **`+17.6%`**; lower loss gives `0%` wake relief to upper).
  - Under 1-motor OEI, naive thrust-matching on the faulted arm demands `2x` thrust (`588.4 N`, `2.43x` power); prioritized **WLS offloading** across the 7 healthy rotors caps peak surviving-rotor thrust at **`60.0%` (`441.5 N`, `1.84x` power)** (theoretical min-peak `53.3%`, `392.4 N`, `1.54x`).
- **1-RC Thevenin Voltage Sag & Headroom Winding:**
  - `V_term = V_oc(SoC) - I * R_0 - v_p`, `dv_p/dt = -v_p/(R_1 C_1) + I/C_1`, `omega_max = (V_term - I_m R_m) / K_e,bus`.
  - Locking `K_e,bus <= 1.36 V*s/rad` (software FOC cap `240 rad/s`) preserves 2-arm `T/W = 1.25` to `358.4 V` and `>= 1.12` down to `338 V`.
- **400 Hz Discrete INDI + `f64` QR/Cholesky WLS (`A in R^{12x8}`):**
  - `Delta_u = WLS(G_1 + G_2, nu_d - (Omega_dot_f + G_2 * Delta_u_prev))`, filtered by identical 8 Hz prewarped 2nd-order Butterworth LPF so phase lags cancel. Never form `float32` normal equations (`cond(H) = 6.385e8` -> negative min eigenvalue); solve `min ||W_v (B u - v)||_2^2 + gamma ||W_u (u - u_p)||_2^2` via QR active-set (`cond = 2.5e4`) or `f64` Cholesky.
- **Same-Arm Dual-Loss CEL (`Option B1b` Active Flat Spin):**
  - Opposite boom drops toward zero thrust; yaw is abandoned into active differential-yaw spin `|Omega_z| = 2.2 rad/s` (21 RPM) with actuator phase lead `Delta_psi = atan(Omega_z * tau_m) = +/- 10.6 deg` (`tau_m = 0.085 s`), holding level tilt `<= 20 deg` and touchdown sink `<= 2.5 / 1.5 / 0.8 m/s`.

---

## Pillar 2: 400 Hz 3-IMU MVS + Chi-Square TMR FDIR

- **Pairwise Discrepancy & Mid-Value Select (MVS):**
  - Across 3 independent IMUs `z_1, z_2, z_3 in R^6`, form pairwise residuals `r_ij = z_i - z_j`. Component-wise median selects the fault-free channel instantaneously on single-lane step/hardover.
- **Gust-Immune `chi2_24` Persistence (`<= 20 ms`, 8 ticks at 400 Hz):**
  - Sliding-window quadratic form `S_ij = sum_{k=1..W} (r_ij,k)^T Sigma_r^{-1} r_ij,k ~ chi^2_{6W}`. If `S_13 > gamma` and `S_23 > gamma` while `S_12 <= gamma`, channel 3 is quarantined cleanly; continuous cross-checking eliminates latent faults (`MOC 5 VTOL.2300`).

---

## Pillar 3: 4D HJI-CBVF Reachability, `DAQP` Safety Filter & `ASTM F3269-21` Arbiter

- **4D Relative-State HJI-VI (`V_f(r, v_r, z, v_z)`):**
  - Solve viscosity PDE `0 = min(l(x) - V_f, dV_f/dt + max_{u in U_f} min_{d in D} grad(V_f) . f(x, u, d))` with matched `_support_U` and `partial_max_magnitudes` for valid Lax-Friedrichs CFL, conservative corner `smax(a,b) - ln(2)/beta`, and degraded cruise speed cap `v_cap,f` for `2M-Cross`.
- **100 Hz Zero-Heap `DAQP` Linear Inequality:**
  - `min_u ||u - u_nom||_2^2` subject to `L_f V_f(x) + L_g V_f(x) u - ||grad_v V_f|| d_max >= -alpha(V_f(x) - eps_grid)` across `K <= 16` simultaneous hazard/CTR rows (`< 2 us` `mmap` `int16` LUT lookup).
- **Anti-Chatter Switching Invariant (`ASTM F3269-21`):**
  - Priority `LATCHED_CEL > LATCHED_CSFL > ACTIVE_CBVF > CF`; reversible CBVF exits iff `t - t_engage >= 1.0 s`, `h(x) >= eps_enter + 0.05`, and `dh/dt > 0`.

---

## Pillar 4: Stratified Atlas Statistics (`log1p`) vs. `ARP4761A` Hardware FTA

- **Exact Clopper-Pearson (`k = 0`) & Zero-Failure Horizon Proof:**
  - `p_U1(k=0, n) = 1 - alpha^(1/n) = -expm1(log(alpha) / n)` (`0.2991%` at `n=1000`, `0.4598%` at `n=650` CSFL, `1.9773%` at `n=150` CEL; strictly isolated denominators!).
  - Exact `10^-9 / FH` horizon via `math.log1p`: `n_min = ceil(log(0.05) / log1p(-1e-9)) = 2,995,732,273` independent flight hours (~341,745 flight years) -> proves Monte Carlo NEVER establishes `10^-9/FH`.
- **Wilson Two-Sided 95% CI:** `(p_hat + z^2/(2n) +/- z * sqrt(p_hat(1-p_hat)/n + z^2/(4n^2))) / (1 + z^2/n)`.
- **Corrected 2oo3 Triplex FCC Cut-Set (`ARP4761A`):**
  - Any `>= 2` failed lanes defeat 2oo3 majority voting: `R_2oo3 = 3 * lambda_fcc^2 * T = 3.00e-8 / FH` at `lambda_fcc = 10^-4/FH` (requires `1oo3` COM-MON self-check or `lambda_fcc <= 1.83e-5/FH` for `<= 10^-9/FH`).

---

## Pillar 5: Advisory AI (`DR-6` / `DR-7`) Strictly Outside RTA

- **Free-Energy OOD & Cell-Local Bayesian SLZ:** `E(x) = -T log sum_c exp(f_c(x)/T)`; cell log-odds `L_t = L_{t-1} + log(p/(1-p))` with explicit OOD variance floor (preventing precision lock-in on new hazards) and cell-local `delta_p = p_W - c_ij` plane covariance (`float32`-safe).
- **Quantitative STL Robustness:** `rho(G_[a,b] phi, x, t) = min_{tau in [t+a, t+b]} rho(phi, x, tau)` via O(1)-amortized `FixedMonotonicWedge` (evicting `head_` before pushing `tail_`).
- **Split Conformal (`KnowNo`, `n_cal = 200, alpha = 0.05`):** Non-conformity quantile `q_hat = Quantile_{ceil((n+1)(1-alpha))/n}({1 - P(y_i | x_i)})` (`q_lvl = 0.955`) triggers Turkish binary clarification during `HOLD_LOITER` whenever prediction set size `|C(x)| > 1`.
