#!/usr/bin/env bash
# Check that the prebuilt darknet binary can actually run on this machine
# before 01_Detect_Images.py is attempted. Linux only.
#
# The binary is a non-PIE x86-64 ELF built with Red Hat GCC 14.3.1 and
# -march=x86-64-v3, so it needs AVX2 + FMA + BMI2. A Xeon E5-2630L v4 has all
# three, but a QEMU guest running the default qemu64 CPU model does not expose
# them and darknet dies with SIGILL.

set -u
cd "$(dirname "$0")"

fail=0
note() { printf '%-12s %s\n' "$1" "$2"; }

echo "== CPU features (need avx2, fma, bmi2) =="
missing=()
for flag in avx2 fma bmi2; do
    if grep -qm1 "\b${flag}\b" /proc/cpuinfo; then
        note "ok" "$flag"
    else
        note "MISSING" "$flag"
        missing+=("$flag")
    fi
done
if [ ${#missing[@]} -ne 0 ]; then
    echo
    echo "  ${missing[*]} not exposed to this machine."
    echo "  If this is a QEMU/KVM guest, restart it with -cpu host (or -cpu Broadwell)."
    echo "  Otherwise darknet must be rebuilt from source with a lower -march."
    fail=1
fi

echo
echo "== Shared libraries =="
if ! ldd ./darknet; then
    echo "  ldd failed - is ./darknet present?"
    fail=1
elif ldd ./darknet | grep -q "not found"; then
    echo
    echo "  Unresolved libraries above. libgomp is the usual one:"
    echo "    sudo dnf install -y libgomp"
    fail=1
fi

echo
echo "== glibc =="
note "required" "<= 2.34 (highest symbol version in the binary)"
note "present" "$(ldd --version | head -1 | awk '{print $NF}')"

echo
echo "== Executable bit =="
if [ -x ./darknet ]; then
    note "ok" "./darknet is executable"
else
    note "MISSING" "run: chmod +x ./darknet"
    fail=1
fi

echo
echo "== Smoke test =="
if ./darknet >/dev/null 2>&1; [ $? -lt 128 ]; then
    note "ok" "binary loads and returns"
else
    note "FAILED" "binary crashed - see CPU features above"
    fail=1
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "PASS - run: python3 01_Detect_Images.py"
else
    echo "FAIL - fix the items above before running 01_Detect_Images.py"
fi
exit "$fail"
