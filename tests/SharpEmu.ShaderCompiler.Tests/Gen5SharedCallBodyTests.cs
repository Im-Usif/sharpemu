// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5SharedCallBodyTests
{
    private static readonly Gen5ShaderProgram Caller = Program(
        MoveScalar(0, 14, 0x2000), MoveScalar(4, 15, 0), MoveScalar(8, 16, 3), MoveScalar(12, 17, 0),
        Sop1(16, "SSwappcB64", 14, Gen5Operand.Scalar(14)),
        EndProgram(20));

    // A callee long enough that a second copy would show in the module size.
    private static Gen5ShaderProgram Callee(uint seed)
    {
        var instructions = new List<Gen5ShaderInstruction>();
        for (uint index = 0; index < 64; index++)
            instructions.Add(Vop2(index * 4, "VAddI32", 1 + index % 8, Operand(seed + index), Gen5Operand.Vector(1 + (index + 1) % 8)));
        instructions.Add(Sop1(64 * 4, "SSetpcB64", 0, Gen5Operand.Scalar(14)));
        return Program([.. instructions]);
    }

    // The module reads its callees' addresses, its records' arguments, its return addresses and a function's own address
    // through the dispatch's data, so the same code and tables placed elsewhere compile to the same module, which a
    // driver cache can reuse.
    [Fact]
    public void LinkedModuleDoesNotDependOnWhereTheCodeLies()
    {
        static byte[] Module(ulong caller, ulong function, ulong first, ulong second)
        {
            var program = Program(Sop1(0, "SSwappcB64", 14, Gen5Operand.Scalar(14)), EndProgram(4)) with { Address = caller };
            var body = Program(Sop1(0, "SGetpcB64", 2, Gen5Operand.Scalar(0)), Vop1(4, "VMovB32", 1, Gen5Operand.Scalar(2)),
                Sop1(8, "SSetpcB64", 0, Gen5Operand.Scalar(14)));
            var linked = Gen5ShaderCallLinker.Link(program,
                [new(0, 14, 16, 14, caller + 4, [new(function, first, body), new(function + 0x400, second, body)])]);
            Assert.Equal([function, function + 0x400], Gen5ShaderCallLinker.CallTargets(linked));
            Assert.Equal([first, second], Gen5ShaderCallLinker.CallEntryArguments(linked));
            Assert.True(Gen5SpirvTranslator.TryCompileProgram(Request(linked), out var shader, out var error), error);
            return shader.Spirv;
        }

        Assert.Equal(Module(0x10000, 0x8000, 0x3000, 0x3010), Module(0x7_0000_0000, 0x5_1234_5000, 0x9_0000_0400, 0x4_0000_0000));
    }

    // A program linked with run-time records marks a call whose callee it does not hold before the call returns at once, and
    // only when the host reads the marks (CallMarks); a program linked per record never marks. A site without callees (the
    // first program of a caller) has no tests, so its blocks are structured; a site with callees keeps the block dispatcher.
    [Theory]
    [InlineData(true, true)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(false, false)]
    public void RuntimeRecordProgramsMarkCallsToCalleesTheyLack(bool callMarks, bool withCallee)
    {
        ShaderCallTarget[] targets = withCallee ? [new ShaderCallTarget(0x2000, 0, Callee(100))] : [];
        var linked = Gen5ShaderCallLinker.Link(Caller, [new(16, 14, 16, 14, 20, targets)], runtimeArguments: true);
        var request = Request(linked, callRecords: true, callMarks: callMarks);
        Assert.Equal(callMarks, request.Bindings.Find(DescriptorBindingKind.CallMarks) is not null);
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out var marked, out var error), error);
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(Request(linked, callRecords: true), out var unmarked, out error), error);
        const uint OpAtomicOr = 241;
        Assert.Equal(callMarks, Ops(marked.Spirv, OpAtomicOr) > Ops(unmarked.Spirv, OpAtomicOr));

        var perRecord = Gen5ShaderCallLinker.Link(Caller, [new(16, 14, 16, 14, 20, [new ShaderCallTarget(0x2000, 0, Callee(100))])]);
        Assert.Null(Request(perRecord, callMarks: true).Bindings.Find(DescriptorBindingKind.CallMarks));
    }

    // The host's bit of a callee is the one the module sets: the steps of CallMarks.Mix on the call's target and the caller's hash.
    [Fact]
    public void CallMarkBitsStayInTheBitsetAndDependOnTheCaller()
    {
        var bits = Enumerable.Range(0, 4096).Select(index => CallMarks.Bit(0x5000_0000_0000UL + (ulong)index * 0x100, Hash)).ToArray();
        Assert.All(bits, bit => Assert.True(bit < CallMarks.Bits));
        Assert.True(bits.Distinct().Count() > 3900, "callee addresses 256 bytes apart spread over the bitset");
        Assert.NotEqual(CallMarks.Bit(0x5000_0000_0000UL, Hash), CallMarks.Bit(0x5000_0000_0000UL, Hash + 1));
    }

    private static int Ops(byte[] spirv, uint opcode)
    {
        var count = 0;
        for (var offset = 20; offset < spirv.Length;)
        {
            var word = BitConverter.ToUInt32(spirv, offset);
            if ((word & 0xFFFF) == opcode) count++;
            offset += (int)(word >> 16) * 4;
        }

        return count;
    }
}
