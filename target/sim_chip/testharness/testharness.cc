#include <iostream>           // For std::cout
#include "Vtestharness.h"     // Verilated generated module
#include "verilated.h"        // Verilator library
#include "verilated_vcd_c.h"  // For VCD tracing (optional)

int main(int argc, char** argv, char** env) {
    Verilated::commandArgs(argc,
                           argv);  // Pass command-line arguments to Verilator
    Vtestharness* top = new Vtestharness;  // Instantiate the top module

    // Variables for tracing
    VerilatedVcdC* tfp = nullptr;
    bool enable_vcd = false;  // Flag to enable VCD tracing

    // Check if the --vcd flag was passed
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--vcd") {
            enable_vcd = true;
            break;
        }
    }

    // Enable tracing if --vcd was passed
    if (enable_vcd) {
        Verilated::traceEverOn(true);  // Enable trace logic
        tfp = new VerilatedVcdC;
        top->trace(tfp, 99);   // Trace 99 levels of hierarchy
        tfp->open("sim.vcd");  // Open the VCD dump file
        std::cout << "VCD trace enabled. Dumping to sim.vcd..." << std::endl;
    }

    // unsigned long main_time = 0;  // Current simulation time
    unsigned long main_time = 0;  // Current simulation time
    // Simulation loop
    while (!Verilated::gotFinish()) {
        top->eval();  // Evaluate the model
        if (tfp)
            tfp->dump(main_time);  // Dump to VCD file if tracing is enabled
        Verilated::timeInc(500);   // Increment the simulation time
        main_time += 500;
    }

    // Clean up
    if (tfp) {
        tfp->close();  // Close VCD file
        delete tfp;
    }

    delete top;
    return 0;
}
