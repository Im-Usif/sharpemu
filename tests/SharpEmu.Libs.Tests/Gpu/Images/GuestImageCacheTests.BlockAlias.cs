// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Tests.Gpu.Buffers;
using Silk.NET.Vulkan;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Images.ImageCacheTestSupport;

namespace SharpEmu.Libs.Tests.Gpu.Images;

public sealed partial class GuestImageCacheTests
{
    // 64x64 BC1: 16x16 blocks of 8 bytes. A compute shader writes the blocks through a 16x16 R16G16B16A16 view of the same memory.
    private const ulong BlockImageBytes = 64 * 64 / 2;

    private static ImageRequest CompressedTexture(ulong address) =>
        LinearRequest(address, BlockImageBytes, Format.BC1RgbaSrgbBlock, GuestPixelFormat.Bc1Srgb, GuestImageType.Color2D, new Extent3D(64, 64, 1), 1, 8, 1);

    private static ImageRequest BlockAliasStorage(ulong address) =>
        AsStorage(LinearRequest(address, BlockImageBytes, Format.R16G16B16A16Uint, GuestPixelFormat.Bits16_16_16_16UInt, GuestImageType.Color2D, new Extent3D(16, 16, 1), 1, 8, 1));

    // An image the guest has not touched for more than the removal window may be deleted by an overlapping request.
    private static void AgeBeyondTheRemovalWindow(CacheHarness harness, ResourceSlotIdentifier imageIdentifier)
    {
        for (var pass = 0; pass < 100 && harness.Scheduler.CurrentTick <= 40; pass++)
        {
            harness.Finish();
        }

        Assert.True(harness.Scheduler.CurrentTick > 40);
        harness.Image(imageIdentifier).LastAccessTick = 0;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FindImage_ABlockAliasOfAGpuWrittenCompressedImageKeepsItsContents(bool aged)
    {
        if (!GatePrerequisites.Ready(_vulkan)) return;
        using var harness = new CacheHarness(_vulkan);
        var address = harness.MapBacked(0x20000, ReadWrite);
        var compressed = CompressedTexture(address);
        var compressedId = harness.Acquire(ref compressed);
        harness.MarkGpuWritten(compressedId);
        if (aged)
        {
            AgeBeyondTheRemovalWindow(harness, compressedId);
        }

        var alias = BlockAliasStorage(address);
        var aliasId = harness.Find(ref alias);

        // The GPU-written blocks only live in the image: the alias has to start from them and replace the compressed image.
        Assert.Equal(new[] { aliasId }, harness.Images.FindImagesInRangeForTest(address, BlockImageBytes, pageOverlap: false));
        Assert.True(harness.Image(aliasId).IsGpuModified);
        Assert.False(harness.Image(aliasId).IsCpuDirty);
    }

    [Fact]
    public void FindImage_ACompressedImageSurvivesABlockAliasRoundTrip()
    {
        if (!GatePrerequisites.Ready(_vulkan)) return;
        using var harness = new CacheHarness(_vulkan);
        var address = harness.MapBacked(0x20000, ReadWrite);
        var compressed = CompressedTexture(address);
        var compressedId = harness.Acquire(ref compressed);
        harness.MarkGpuWritten(compressedId);
        AgeBeyondTheRemovalWindow(harness, compressedId);

        var alias = BlockAliasStorage(address);
        var aliasId = harness.Find(ref alias);
        harness.MarkGpuWritten(aliasId);
        AgeBeyondTheRemovalWindow(harness, aliasId);

        var sampled = CompressedTexture(address);
        var sampledId = harness.Find(ref sampled);

        Assert.Equal(new[] { sampledId }, harness.Images.FindImagesInRangeForTest(address, BlockImageBytes, pageOverlap: false));
        Assert.True(harness.Image(sampledId).Description.IsBlock);
        Assert.True(harness.Image(sampledId).IsGpuModified);
    }
}
