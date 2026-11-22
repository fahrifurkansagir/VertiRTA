include(FetchContent)

cmake_policy(PUSH)
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()
FetchContent_Declare(
  daqp
  GIT_REPOSITORY https://github.com/darnstrom/daqp.git
  GIT_TAG v0.10.0
  GIT_SHALLOW TRUE
)
FetchContent_GetProperties(daqp)
if(NOT daqp_POPULATED)
  FetchContent_Populate(daqp)
endif()
cmake_policy(POP)

# Generate a MISRA-C / DO-178C acyclic-callgraph copy of DAQP v0.10.0 auxiliary.c:
# upstream daqp_pivot_last() mutually recurses with daqp_remove_constraint() and
# daqp_add_constraint_keep_slack(); replace with bounded iterative helpers (zero recursion).
set(_daqp_aux_orig "${daqp_SOURCE_DIR}/src/auxiliary.c")
set(_daqp_aux_norec "${CMAKE_CURRENT_BINARY_DIR}/daqp_auxiliary_norec.c")
file(READ "${_daqp_aux_orig}" _daqp_aux_src)

set(_daqp_old_1 "void daqp_remove_constraint(DAQPWorkspace* work, const int rm_ind){
    int i;
    // Update data structures
    DAQP_SET_INACTIVE(work->WS[rm_ind]);
    daqp_update_LDL_remove(work,rm_ind);
    (work->n_active)--;

    for(i=rm_ind;i<work->n_active;i++){
        work->WS[i] = work->WS[i+1];
        work->lam[i] = work->lam[i+1];
    }
    // Can only reuse work less than the ind that was removed
    if(rm_ind < work->reuse_ind)
        work->reuse_ind = rm_ind;

    // Check if the removal lead to singularity (can happen due to numerics).
    if(work->n_active > 0 && work->D[work->n_active-1] < work->settings->sing_tol)
        work->sing_ind = work->n_active-1;
    else{ // Pivot for improved numerics
        daqp_pivot_last(work);
    }
}
// Add a constraint, keeping the slack state that is marked in sense
static void daqp_add_constraint_keep_slack(DAQPWorkspace *work,
        const int add_ind, c_float lam){
    // Update data structures
    DAQP_SET_ACTIVE(add_ind);
    const c_float rho = DAQP_IS_SOFT(add_ind) && DAQP_IS_SLACK_FREE(add_ind)
        ? daqp_soft_rho(work,add_ind) : 0;
    daqp_update_LDL_add(work,add_ind,rho);
    work->WS[work->n_active] = add_ind;
    work->lam[work->n_active] = lam;
    work->n_active++;

    // Pivot for improved numerics
    daqp_pivot_last(work);
}")

set(_daqp_new_1 "static void daqp_remove_nopivot(DAQPWorkspace* work, const int rm_ind){
    int i;
    DAQP_SET_INACTIVE(work->WS[rm_ind]);
    daqp_update_LDL_remove(work,rm_ind);
    (work->n_active)--;
    for(i=rm_ind;i<work->n_active;i++){
        work->WS[i] = work->WS[i+1];
        work->lam[i] = work->lam[i+1];
    }
    if(rm_ind < work->reuse_ind)
        work->reuse_ind = rm_ind;
    if(work->n_active > 0 && work->D[work->n_active-1] < work->settings->sing_tol)
        work->sing_ind = work->n_active-1;
}
static void daqp_add_keep_slack_nopivot(DAQPWorkspace *work,
        const int add_ind, c_float lam){
    DAQP_SET_ACTIVE(add_ind);
    const c_float rho = DAQP_IS_SOFT(add_ind) && DAQP_IS_SLACK_FREE(add_ind)
        ? daqp_soft_rho(work,add_ind) : 0;
    daqp_update_LDL_add(work,add_ind,rho);
    work->WS[work->n_active] = add_ind;
    work->lam[work->n_active] = lam;
    work->n_active++;
}
void daqp_remove_constraint(DAQPWorkspace* work, const int rm_ind){
    daqp_remove_nopivot(work,rm_ind);
    if(work->sing_ind == DAQP_EMPTY_IND){
        daqp_pivot_last(work);
    }
}
static void daqp_add_constraint_keep_slack(DAQPWorkspace *work,
        const int add_ind, c_float lam){
    daqp_add_keep_slack_nopivot(work,add_ind,lam);
    daqp_pivot_last(work);
}")

set(_daqp_old_2 "void daqp_pivot_last(DAQPWorkspace *work){
    const int rm_ind = work->n_active-2;
    if(work->n_active > 1 && work->sing_ind == DAQP_EMPTY_IND &&
            work->D[rm_ind] < work->settings->pivot_tol && // element in D small enough
            work->D[rm_ind] < work->D[work->n_active-1]){ // element in D smallar than neighbor
        const int ind_old = work->WS[rm_ind];
        // Ensure that binaries never swap order (since this order is exploited)
        if(DAQP_IS_BINARY(ind_old) && DAQP_IS_BINARY(work->WS[work->n_active-1])) return;
        if(work->bnb != NULL && rm_ind < work->bnb->n_clean) return;

        c_float lam_old = work->lam[rm_ind];
        daqp_remove_constraint(work,rm_ind); // pivot_last might be recursively called here

        if(work->sing_ind!=DAQP_EMPTY_IND) return; // Abort if D becomes singular

        // Reordering only: the slack state of the constraint is unchanged
        daqp_add_constraint_keep_slack(work,ind_old,lam_old);
    }
}")

set(_daqp_new_2 "void daqp_pivot_last(DAQPWorkspace *work){
    int iter;
    for(iter = 0; iter < work->n_active; ++iter){
        const int rm_ind = work->n_active-2;
        if(!(work->n_active > 1 && work->sing_ind == DAQP_EMPTY_IND &&
                work->D[rm_ind] < work->settings->pivot_tol &&
                work->D[rm_ind] < work->D[work->n_active-1])) break;
        const int ind_old = work->WS[rm_ind];
        if(DAQP_IS_BINARY(ind_old) && DAQP_IS_BINARY(work->WS[work->n_active-1])) break;
        if(work->bnb != NULL && rm_ind < work->bnb->n_clean) break;
        const c_float lam_old = work->lam[rm_ind];
        daqp_remove_nopivot(work,rm_ind);
        if(work->sing_ind != DAQP_EMPTY_IND) break;
        daqp_add_keep_slack_nopivot(work,ind_old,lam_old);
    }
}")

string(FIND "${_daqp_aux_src}" "${_daqp_old_1}" _daqp_idx_1)
string(FIND "${_daqp_aux_src}" "${_daqp_old_2}" _daqp_idx_2)
if(_daqp_idx_1 EQUAL -1 OR _daqp_idx_2 EQUAL -1)
  message(FATAL_ERROR "DAQP v0.10.0 auxiliary.c recursion-elimination patch anchor not found!")
endif()
string(REPLACE "${_daqp_old_1}" "${_daqp_new_1}" _daqp_aux_src "${_daqp_aux_src}")
string(REPLACE "${_daqp_old_2}" "${_daqp_new_2}" _daqp_aux_src "${_daqp_aux_src}")
file(WRITE "${_daqp_aux_norec}" "${_daqp_aux_src}")

add_library(vertirta_flight_core STATIC
  src/frames.cpp
  src/tmr_voter.cpp
  src/coax_envelope.cpp
  src/indi_wls.cpp
  src/motor_fdi.cpp
  src/daqp_qp.cpp
  src/vrta_recorder.cpp
  src/sim_plant_6dof.cpp
  ${daqp_SOURCE_DIR}/src/daqp.c
  ${_daqp_aux_norec}
  ${daqp_SOURCE_DIR}/src/factorization.c
)
add_library(vertirta::flight_core ALIAS vertirta_flight_core)

target_compile_features(vertirta_flight_core PUBLIC cxx_std_17)
target_include_directories(vertirta_flight_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${daqp_SOURCE_DIR}/include
)

# Apply strict warnings to our C++17 flight_core sources; keep third-party C99 DAQP warning-clean.
set_source_files_properties(
  src/frames.cpp
  src/tmr_voter.cpp
  src/coax_envelope.cpp
  src/indi_wls.cpp
  src/motor_fdi.cpp
  src/daqp_qp.cpp
  src/vrta_recorder.cpp
  src/sim_plant_6dof.cpp
  PROPERTIES COMPILE_OPTIONS
    "-Wall;-Wextra;-Wpedantic;-Werror;-Wshadow;-Wconversion;-Wuninitialized"
)

target_compile_options(vertirta_flight_core PRIVATE
  -ffunction-sections
  -fdata-sections
  $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions -fno-rtti -fno-threadsafe-statics>
  $<$<BOOL:${VERTIRTA_ENABLE_STRICT_FP}>:${VERTIRTA_STRICT_FP_FLAGS}>
)
