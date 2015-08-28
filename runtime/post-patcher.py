#!/usr/bin/env python3
import sys, re, os
rttype, basename = sys.argv[1:]

if rttype in ("nvvm", "spir"):
    # we need to patch
    result = []
    filename = basename+"."+rttype
    if os.path.isfile(filename):
        with open(filename) as f:
            for line in f:
                # patch to opaque identity functions
                if rttype=="spir":
                    m = re.match('^declare cc75 (.*) @(magic_.*_id)\((.*)\)\n$', line)
                else:
                    m = re.match('^declare (.*) @(magic_.*_id)\((.*)\)\n$', line)
                if m is not None:
                    ty1, fname, ty2 = m.groups()
                    assert ty1 == ty2, "Argument and return types of magic IDs must match"
                    print("Patching magic ID {0}".format(fname))
                    # emit definition instead
                    if rttype=="spir":
                        result.append('define cc75 {0} @{1}({0} %name) {{\n'.format(ty1, fname))
                    else:
                        result.append('define {0} @{1}({0} %name) {{\n'.format(ty1, fname))
                    result.append('  ret {0} %name\n'.format(ty1))
                    result.append('}\n')
                # get rid of attributes
                elif 'attributes #' in line:
                    print("Removing attribute declarations")
                    # ignore this line
                    pass
                # emit correct datalayout for llvm 3.6
                elif 'target datalayout = "e-i32' in line:
                    if rttype=="spir":
                        result.append('target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"\n')
                    else:
                        result.append('target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"\n')
                elif 'target datalayout = "e-i64' in line:
                    if rttype=="spir":
                        result.append('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"\n')
                    else:
                        result.append('target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"\n')
                # it's a normal line, keep it but apply substitutions
                else:
                    # get rid of attributes at end of line
                    line = re.sub('#[0-9]+', '', line)
                    # add metadata for llvm 3.6
                    line = re.sub(r'\A!([0-9]+) = !', r'!\1 = metadata !', line)
                    line = re.sub(r', !"kernel",', r', metadata !"kernel",', line)
                    line = re.sub(r', !"texture",', r', metadata !"texture",', line)
                    result.append(line)
        # we have the patched thing, write it
        with open(filename, "w") as f:
            for line in result:
                f.write(line)

if rttype in ("cuda"):
    # we need to patch
    result = []
    filename = basename+".cu"
    if os.path.isfile(filename):
        with open(filename) as f:
            for line in f:
                # patch to opaque identity functions
                m = re.match('^__device__ (.*) (magic_.*_id)\((.*)\);\n$', line)
                if m is not None:
                    ty1, fname, ty2 = m.groups()
                    assert ty1 == ty2, "Argument and return types of magic IDs must match"
                    print("Patching magic ID {0}".format(fname))
                    # emit definition instead
                    result.append('__device__ {0} {1}({0} name) {{\n'.format(ty1, fname))
                    result.append('  return name;\n')
                    result.append('}\n')
                else:
                    result.append(line)
        # we have the patched thing, write it
        with open(filename, "w") as f:
            for line in result:
                f.write(line)

if rttype in ("opencl"):
    # we need to patch
    result = []
    filename = basename+".cl"
    if os.path.isfile(filename):
        with open(filename) as f:
            for line in f:
                # patch to opaque identity functions
                m = re.match('^(?!.*=)(.*) (magic_.*_id)\((.*)\);\n$', line)
                if m is not None:
                    ty1, fname, ty2 = m.groups()
                    assert ty1 == ty2, "Argument and return types of magic IDs must match"
                    print("Patching magic ID {0}".format(fname))
                    # emit definition instead
                    result.append('{0} {1}({0} name) {{\n'.format(ty1, fname))
                    result.append('  return name;\n')
                    result.append('}\n')
                else:
                    result.append(line)
        # we have the patched thing, write it
        with open(filename, "w") as f:
            for line in result:
                f.write(line)

# another pass to add the ldg, minmax and consorts to the nvvm file
nvvm_defs = {
  "ldg4_f32" : """define <4 x float> @ldg4_f32(<4 x float>* %addr) {
    %1 = call {float, float, float, float} asm "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f, l" (<4 x float>* %addr)
    %2 = extractvalue {float, float, float, float} %1, 0
    %3 = extractvalue {float, float, float, float} %1, 1
    %4 = extractvalue {float, float, float, float} %1, 2
    %5 = extractvalue {float, float, float, float} %1, 3
    %6 = insertelement <4 x float> undef, float %2, i32 0
    %7 = insertelement <4 x float> %6, float %3, i32 1
    %8 = insertelement <4 x float> %7, float %4, i32 2
    %9 = insertelement <4 x float> %8, float %5, i32 3
    ret <4 x float> %9
}
""",
  "ldg4_i32" : """define <4 x i32> @ldg4_i32(<4 x i32>* %addr) {
    %1 = call {i32, i32, i32, i32} asm "ld.global.nc.v4.s32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r, l" (<4 x i32>* %addr)
    %2 = extractvalue {i32, i32, i32, i32} %1, 0
    %3 = extractvalue {i32, i32, i32, i32} %1, 1
    %4 = extractvalue {i32, i32, i32, i32} %1, 2
    %5 = extractvalue {i32, i32, i32, i32} %1, 3
    %6 = insertelement <4 x i32> undef, i32 %2, i32 0
    %7 = insertelement <4 x i32> %6, i32 %3, i32 1
    %8 = insertelement <4 x i32> %7, i32 %4, i32 2
    %9 = insertelement <4 x i32> %8, i32 %5, i32 3
    ret <4 x i32> %9
}
""",
  "maxmax" : """define i32 @maxmax(i32 %a, i32 %b, i32 %c) {
    %1 = call i32 asm "vmax.s32.s32.s32.max $0, $1, $2, $3;", "=r, r, r, r" (i32 %a, i32 %b, i32 %c)
    ret i32 %1
}
""",
  "minmin" : """define i32 @minmin(i32 %a, i32 %b, i32 %c) {
    %1 = call i32 asm "vmin.s32.s32.s32.min $0, $1, $2, $3;", "=r, r, r, r" (i32 %a, i32 %b, i32 %c)
    ret i32 %1
}
""",
  "minmax" : """define i32 @minmax(i32 %a, i32 %b, i32 %c) {
    %1 = call i32 asm "vmin.s32.s32.s32.max $0, $1, $2, $3;", "=r, r, r, r" (i32 %a, i32 %b, i32 %c)
    ret i32 %1
}
""",
  "maxmin" : """define i32 @maxmin(i32 %a, i32 %b, i32 %c) {
    %1 = call i32 asm "vmax.s32.s32.s32.min $0, $1, $2, $3;", "=r, r, r, r" (i32 %a, i32 %b, i32 %c)
    ret i32 %1
}
"""
}

if rttype == "nvvm":
    result = []
    filename = basename+".nvvm"
    if os.path.isfile(filename):
        with open(filename) as f:
            for line in f:
                matched = False

                for (func, code) in nvvm_defs.iteritems():
                    m = re.match('^declare (.*) (@' + func + ')\((.*)\)\n$', line)
                    if m is not None:
                        result.append(code)
                        matched = True
                        break

                if not matched:
                    result.append(line)

        with open(filename, "w") as f:
            for line in result:
                f.write(line)

