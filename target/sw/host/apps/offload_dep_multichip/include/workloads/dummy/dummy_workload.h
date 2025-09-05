#include "dummy_data.h"
#include "libbingo/bingo_api.h"
#include "host.h"
// Only the main chiplet will run this function
// Since the main memory will be copied exactly to all the chiplets
// The lower address of the dev_array will be the same
// Only the upper 48-40 chiplet id prefix will be different
void __workload_multichip_dummy(bingo_task_t **task_list, uint32_t *num_tasks_ptr) {
    *num_tasks_ptr = 4;
    // Get the kernel function address
    check_kernel_tab_ready();
    uint32_t dummy_func_addr = get_device_function("__snax_kernel_dummy");
    if (dummy_func_addr == SNAX_SYMTAB_END_FN_ADDR) {
        printf("Error: Kernel symbol lookup failed!\r\n");
    }
    // Register the tasks
    // printf("[Host] Start to Register tasks\r\n");
    bingo_task_t *task_dummy_chip0_cluster0 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_chip0_cluster0));
    bingo_task_t *task_dummy_chip0_cluster1 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_chip0_cluster1));
    bingo_task_t *task_dummy_chip1_cluster0 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_chip1_cluster0));
    bingo_task_t *task_dummy_chip1_cluster1 = bingo_task_create(dummy_func_addr, (uint32_t)(uintptr_t)(&dummy_args_chip1_cluster1));
    // Set the task dependency
    //      c0_cl0
    //           |
    //           v
    //  ----c1_cl1
    //  |        |
    //  v        v
    // c0_cl1   c1_cl0
    // So the c0_cl0 will start to run
    // Then the c1_cl1 will run after
    // Then the c0_cl0 and  c1_cl0 will run in parallel
    // We set the dependency from the child to the parent
    bingo_task_add_depend(task_dummy_chip0_cluster1, task_dummy_chip1_cluster1);
    bingo_task_add_depend(task_dummy_chip1_cluster0, task_dummy_chip1_cluster1);
    bingo_task_add_depend(task_dummy_chip1_cluster1, task_dummy_chip0_cluster0);
    //Set the assigned chiplet id and cluster id
    task_dummy_chip0_cluster0->assigned_chip_id = 0;
    task_dummy_chip0_cluster0->assigned_cluster_id = 0;
    task_dummy_chip0_cluster1->assigned_chip_id = 0;
    task_dummy_chip0_cluster1->assigned_cluster_id = 1;
    task_dummy_chip1_cluster0->assigned_chip_id = 1;
    task_dummy_chip1_cluster0->assigned_cluster_id = 0;
    task_dummy_chip1_cluster1->assigned_chip_id = 1;
    task_dummy_chip1_cluster1->assigned_cluster_id = 1;

    // Set up the task list
    task_list[0] = task_dummy_chip0_cluster0;
    task_list[1] = task_dummy_chip0_cluster1;
    task_list[2] = task_dummy_chip1_cluster0;
    task_list[3] = task_dummy_chip1_cluster1;
}