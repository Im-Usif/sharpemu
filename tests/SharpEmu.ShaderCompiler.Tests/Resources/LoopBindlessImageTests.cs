// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

// A compute pass walks a table of records. Each record holds a bindless heap index; the loop body fetches the
// heap descriptor with S_BUFFER_LOAD and samples it. The material key is selected by the loop counter, which every
// lane shares, rather than by a lane value read through v_readfirstlane.
public sealed class LoopBindlessImageTests
{
    private const uint Format32x4Float = 77;
    private const uint ImageType3D = 10;
    private const uint IdentitySwizzle = 0xFAC;
    private const uint RecordStride = 224;
    private const uint Records = 8;
    private const uint KeyOffset = 4;
    private const uint ThreeDimensions = 2;

    private static short Back(uint from, uint to) => (short)(((int)to - (int)from - 4) / 4);

    // The material table is s[0:3] and the heap s[4:7]. The descriptor s[16:23] is fetched in the loop body, or
    // before an inner loop that samples it; then its dwords reach the image as loop-invariant phis.
    private static Gen5ShaderProgram LoopProgram(bool descriptorBeforeInnerLoop, bool keyFromMaterialTable = true, bool storeResult = false)
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        void Add(Gen5ShaderInstruction instruction) { instructions.Add(instruction); pc += 8; }

        Add(MoveScalar(pc, 9, 0));
        var outer = pc;
        Add(Sop2(pc, "SMulI32", 10, Gen5Operand.Scalar(9), Operand(RecordStride)));
        Add(Sop2(pc, "SAddU32", 11, Gen5Operand.Scalar(10), Operand(KeyOffset)));
        if (keyFromMaterialTable)
        {
            Add(ScalarBufferLoad(pc, 0, destination: 12, dynamicOffsetRegister: 11));
        }
        else
        {
            Add(Sop2(pc, "SAddU32", 12, Gen5Operand.Scalar(11), Operand(1)));
        }

        Add(Sop2(pc, "SLshlB32", 13, Gen5Operand.Scalar(12), Operand(5)));
        Add(ScalarBufferLoad(pc, 4, destination: 16, count: 8, dynamicOffsetRegister: 13));
        for (uint index = 0; index < 4; index++)
        {
            Add(MoveScalar(pc, 24 + index, 0));
        }

        if (descriptorBeforeInnerLoop)
        {
            Add(MoveScalar(pc, 30, 0));
            var inner = pc;
            Add(Image(pc, "ImageSampleLz", 16, 24, dimension: ThreeDimensions, dmask: 1));
            Add(Sop2(pc, "SAddU32", 30, Gen5Operand.Scalar(30), Operand(1)));
            Add(Sopc(pc, "SCmpLgU32", Gen5Operand.Scalar(30), Operand(2)));
            Add(Branch(pc, "SCbranchScc1", Back(pc, inner)));
        }
        else
        {
            Add(Image(pc, "ImageSampleLz", 16, 24, dimension: ThreeDimensions, dmask: 1));
        }

        if (storeResult)
        {
            Add(BufferAccess(pc, "BufferStoreDword", 52, 0, 1, vectorData: 4));
        }

        Add(Sop2(pc, "SAddU32", 9, Gen5Operand.Scalar(9), Operand(1)));
        Add(Sopc(pc, "SCmpLgU32", Gen5Operand.Scalar(9), Operand(Records)));
        Add(Branch(pc, "SCbranchScc1", Back(pc, outer)));
        Add(EndProgram(pc));
        return Program([.. instructions]);
    }

    // The same fetch with a selector that is one user data dword instead of a loop counter: the host can evaluate
    // that descriptor directly.
    private static Gen5ShaderProgram UserDataSelectorProgram()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        void Add(Gen5ShaderInstruction instruction) { instructions.Add(instruction); pc += 8; }

        Add(Sop2(pc, "SMulI32", 10, Gen5Operand.Scalar(8), Operand(RecordStride)));
        Add(Sop2(pc, "SAddU32", 11, Gen5Operand.Scalar(10), Operand(KeyOffset)));
        Add(ScalarBufferLoad(pc, 0, destination: 12, dynamicOffsetRegister: 11));
        Add(Sop2(pc, "SLshlB32", 13, Gen5Operand.Scalar(12), Operand(5)));
        Add(ScalarBufferLoad(pc, 4, destination: 16, count: 8, dynamicOffsetRegister: 13));
        for (uint index = 0; index < 4; index++)
        {
            Add(MoveScalar(pc, 24 + index, 0));
        }

        Add(Image(pc, "ImageSampleLz", 16, 24, dimension: ThreeDimensions, dmask: 1));
        Add(EndProgram(pc));
        return Program([.. instructions]);
    }

    private static uint[] Volume(uint address) =>
        [address, Format32x4Float << 20, 3 | (3 << 14), IdentitySwizzle | (ImageType3D << 28), 0, 0, 0, 0];

    // Eight records whose keys are 1, 2, 1, 3, 0, 0, 0, 0; heap entry 0 stays empty.
    private static (uint[] UserData, TestWordMemory Memory) Inputs3D()
    {
        var userData = new uint[64];
        uint[] tables = [0x1000, RecordStride << 16, Records, 0, 0x2000, 32 << 16, 4, 0, 1];
        tables.CopyTo(userData, 0);
        userData[54] = 64;
        var memory = ResourceTrackerTests.LinearMemory();
        ResourceTrackerTests.WriteImage(memory, 0x2020, Volume(0x20));
        ResourceTrackerTests.WriteImage(memory, 0x2040, Volume(0x30));
        ResourceTrackerTests.WriteImage(memory, 0x2060, Volume(0x40));
        uint[] keys = [1, 2, 1, 3, 0, 0, 0, 0];
        for (uint record = 0; record < Records; record++)
        {
            memory.At(0x1000 + record * RecordStride + KeyOffset) = keys[record];
        }

        return (userData, memory);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LoopCounterSelector_BecomesAnIndirectImage(bool descriptorBeforeInnerLoop)
    {
        var plan = Extract(LoopProgram(descriptorBeforeInnerLoop));

        Assert.Single(plan.Info.Images);
        var source = plan.DescriptorSources[(int)plan.Info.Images[0].Source];
        Assert.NotNull(source.IndirectImage);
        Assert.Single(plan.IndirectImages);
        Assert.Equal(ScalarValueKind.ScalarBufferWord, plan.IndirectImages[0].Key.Kind);
        Assert.Equal(RecordStride, source.IndirectImage!.SelectorStride);
        Assert.Equal(KeyOffset, source.IndirectImage.SelectorOffset);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LoopCounterSelector_MaterializesEveryHeapDescriptorTheTableNames(bool descriptorBeforeInnerLoop)
    {
        var plan = Extract(LoopProgram(descriptorBeforeInnerLoop));
        var (userData, memory) = Inputs3D();
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();

        Assert.True(ResourceMaterializer.Materialize(plan, Inputs(userData, readCleanMemory: memory.Read), ref snapshot, ref specialization));

        // The empty entry (key 0) and the three volumes the records name.
        Assert.Equal(4, snapshot.Images.Length);
        foreach (var address in new uint[] { 0x20, 0x30, 0x40 })
        {
            Assert.Contains(snapshot.Images, image => image.SequenceEqual(Volume(address)));
        }

        var applied = ResourceMaterializer.ApplyTo(plan, specialization);
        Assert.Equal(4, applied.Info.Images.Count);
        Assert.All(applied.Info.Images, image => Assert.Equal(0u, image.IndirectRoot));
        Assert.NotEqual(0u, applied.Info.Images[0].IndirectSearchIterations);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LoopCounterSelector_CompilesWithCandidateSelection(bool descriptorBeforeInnerLoop)
    {
        var plan = Extract(LoopProgram(descriptorBeforeInnerLoop, storeResult: true));
        var (userData, memory) = Inputs3D();
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs(userData, readCleanMemory: memory.Read), ref snapshot, ref specialization));
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        Assert.Equal(4, resources.Info.Images.Count);
        var layout = BindingLayout.Allocate(
            resources.Info,
            BindingLayout.CollectUserDataRegisters(plan.Graph.Program, 0, 64),
            false,
            ShaderCompileRequest.RequiresFlattenedTable(plan, resources),
            false);

        Assert.True(Gen5SpirvTranslator.TryCompileProgram(new ShaderCompileRequest(plan, resources, layout), out var shader, out var error), error);
        Assert.Contains((ushort)SpirvOp.Select, new SpirvModuleInspector(shader.Spirv).Opcodes);
    }

    [Fact]
    public void UserDataSelector_KeepsTheDirectDescriptor()
    {
        var plan = Extract(UserDataSelectorProgram());

        Assert.Single(plan.Info.Images);
        Assert.Null(plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage);
        Assert.Empty(plan.IndirectImages);
    }

    [Fact]
    public void LoopWithoutAMaterialKey_StaysANullImage()
    {
        var plan = Extract(LoopProgram(false, keyFromMaterialTable: false));

        Assert.Single(plan.Info.Images);
        var source = plan.DescriptorSources[(int)plan.Info.Images[0].Source];
        Assert.Null(source.IndirectImage);
        Assert.Empty(plan.IndirectImages);
        Assert.All(source.Dwords, dword => Assert.True(dword.IsConstant && dword.ConstantU32 == 0));
    }
}
