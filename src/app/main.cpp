#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include <boost/program_options.hpp>
#include <boost/array.hpp>

#include "anydsl/literal.h"
#include "anydsl/lambda.h"
#include "anydsl/primop.h"
#include "anydsl/world.h"
#include "impala/parser.h"
#include "impala/ast.h"
#include "impala/sema.h"
#include "impala/dump.h"
#include "impala/emit.h"
#include "impala/type.h"
#include "impala/init.h"

//------------------------------------------------------------------------------

using namespace anydsl;
using namespace std;
namespace po = boost::program_options;

typedef vector<string> Names;

//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    try {
        if (argc < 1)
            throw logic_error("bad number of arguments");

        string prgname = argv[0];
        Names infiles;
        string outfile = "-";
        string emittype;
        bool help, emit_air, emit_ast, emit_dot, fancy = false;

        // specify options
        po::options_description desc("Usage: " + prgname + " [options] file...");
        desc.add_options()
        ("help,h",      po::bool_switch(&help),                     "produce this help message")
        ("emit-air",    po::bool_switch(&emit_air),                 "emit textual AIR representation of impala program")
        ("emit-ast",    po::bool_switch(&emit_ast),                 "emit AST of impala program")
        ("emit-dot",    po::bool_switch(&emit_dot),                 "emit dot, arg={air|llvm}")
        ("fancy,f",     po::bool_switch(&fancy),                    "use fancy output")
        ("outfile,o",   po::value(&outfile)->default_value("-"),    "specifies output file")
        ("infile,i",    po::value(&infiles),                        "input file");

        // positional options, i.e., input files
        po::positional_options_description pos_desc;
        pos_desc.add("infile", -1);

        // do cmdline parsing
        po::command_line_parser clp(argc, argv);
        clp.options(desc);
        clp.positional(pos_desc);
        po::variables_map vm;

        po::store(clp.run(), vm);
        po::notify(vm);

        if (infiles.empty() && !help)
            throw po::invalid_syntax(po::invalid_syntax::missing_parameter, "infile");

        if (help) {
            desc.print(cout);
            return EXIT_SUCCESS;
        }

        ofstream ofs;
        if (outfile != "-") {
            ofs.open(outfile.c_str());
            ofs.exceptions(istream::badbit);
        }
        //ostream& out = ofs.is_open() ? ofs : cout;

        const char* filename = infiles[0].c_str();
        ifstream file(filename);

        impala::Init init;

        anydsl::AutoPtr<const impala::Prg> p(impala::parse(init.types, file, filename));
        check(init.types, p);

        if (emit_ast)
            dump(p, fancy);
        if (emit_dot)
            ANYDSL_NOT_IMPLEMENTED;
        if (emit_air) {
            emit(init.world, p);
            init.world.dump(fancy);
        }
        
        return EXIT_SUCCESS;
    } catch (exception const& e) {
        cerr << e.what() << endl;
        return EXIT_FAILURE;
    } catch (...) {
        cerr << "unknown exception" << endl;
        return EXIT_FAILURE;
    }
}
