#include <getopt.h>
#include <filesystem>
#include <iostream>        // For std::cout
#include "Vtestharness.h"  // Verilated generated module
#include "Vtestharness__Dpi.h"
#include "verilated.h"        // Verilator library
#include "verilated_vcd_c.h"  // For VCD tracing (optional)

// Function to print help message
void print_help() {
    std::cout
        << "HeMAIA Verilator Simulation Driver\n"
        << "     -h :\t\t\t Show this help message\n"
        << "     --vcd :\t\t\t Enable VCD trace dumping\n"
        << "     --disable-tracing : \t Disable DASM tracing\n"
        << "     --prefix-trace <prefix>: \t Prefix DASM files with prefix\n"
        << "                              \t(cannot be used together with "
           "--disable-tracing)\n";
}

// Global variables
bool WRAPPER_disable_tracing = false;
std::string WRAPPER_trace_prefix;

// DPI calls
svBit disable_tracing() {
    // function to enable/disable tracers
    return WRAPPER_disable_tracing;
}

const char* get_trace_file_prefix() {
    if (WRAPPER_trace_prefix.empty()) {
        // Use the standard prefix, and create a logs directory if necessary
        std::string foldername = "logs/";
        std::filesystem::create_directories(foldername);
        static std::string log_file_name = "logs/";
        return log_file_name.c_str();
    }
    // Return the one parsed from the command line otherwise
    else {
        return WRAPPER_trace_prefix.c_str();
    }
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc,
                           argv);  // Pass command-line arguments to Verilator
    Vtestharness* top = new Vtestharness;  // Instantiate the top module

    // Variables for options
    bool enable_vcd = false;

    // Define the long options
    static struct option long_options[] = {
        {"vcd", no_argument, nullptr, 0},
        {"disable-tracing", no_argument, nullptr, 1},
        {"prefix-trace", required_argument, nullptr, 2},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}  // Terminate the option array
    };

    // Parse the command-line arguments
    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "h", long_options, &option_index)) !=
           -1) {
        switch (c) {
            case 0:  // --vcd
                enable_vcd = true;
                break;
            case 1:  // --disable-tracing
                WRAPPER_disable_tracing = true;
                break;
            case 2:  // --trace-prefix
                WRAPPER_trace_prefix = optarg;
                break;
            case 'h':  // -h or --help
                print_help();
                return 0;
            default:
                std::cerr << "Unknown option. Use -h for help.\n";
                return 1;
        }
    }

    // Validate conflicting options
    if (WRAPPER_disable_tracing && !WRAPPER_trace_prefix.empty()) {
        std::cerr << "Error: --trace-prefix cannot be used together with "
                     "--disable-tracing\n";
        return 1;
    }

    // Handle VCD tracing
    VerilatedVcdC* tfp = nullptr;
    if (enable_vcd) {
        Verilated::traceEverOn(true);  // Enable trace logic
        tfp = new VerilatedVcdC;
        top->trace(tfp, 99);   // Trace 99 levels of hierarchy
        tfp->open("sim.vcd");  // Open the VCD dump file
        std::cout << "VCD trace enabled. Dumping to sim.vcd...\n";
    }

    // Additional logging based on other options
    if (WRAPPER_disable_tracing) {
        std::cout << "DASM tracing is disabled.\n";
    } else if (!WRAPPER_trace_prefix.empty()) {
        std::cout << "DASM tracing enabled with prefix: "
                  << WRAPPER_trace_prefix << "\n";
    }

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
