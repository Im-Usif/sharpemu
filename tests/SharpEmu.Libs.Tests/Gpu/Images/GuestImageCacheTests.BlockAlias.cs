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

    private const uint BlockArrayLayers = 4;

    private static ImageRequest CompressedArray(ulong address) =>
        LinearRequest(address, BlockImageBytes * BlockArrayLayers, Format.BC1RgbaSrgbBlock, GuestPixelFormat.Bc1Srgb, GuestImageType.Color2D,
            new Extent3D(64, 64, 1), BlockArrayLayers, 8, 1);

    // A shader reads the blocks of the array through an uncompressed view of the given layers (a copy of streamed mips).
    private static ImageRequest BlockAliasArray(ulong address, uint viewLayers)
    {
        var request = LinearRequest(address, BlockImageBytes * BlockArrayLayers, Format.R16G16B16A16Uint, GuestPixelFormat.Bits16_16_16_16UInt,
            GuestImageType.Color2D, new Extent3D(16, 16, 1), BlockArrayLayers, 8, 1);
        request.View = request.View with { LayerCount = viewLayers, Type = viewLayers > 1 ? ImageViewType.Type2DArray : ImageViewType.Type2D };
        return request;
    }

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

    // Vulkan views the blocks of a compressed image in an uncompressed format one level and one layer at a time
    // (VK_KHR_maintenance2). A view of several layers is undefined: the driver lays its layers out as a one-level
    // image and the shader reads blocks of other layers. The blocks move to an uncompressed image instead.
    [Fact]
    public void FindImage_AnUncompressedViewOfSeveralLayersOfACompressedArrayIsNotAViewOfIt()
    {
        if (!GatePrerequisites.Ready(_vulkan)) return;
        using var harness = new CacheHarness(_vulkan);
        var address = harness.MapBacked(0x20000, ReadWrite);
        var compressed = CompressedArray(address);
        var compressedId = harness.Acquire(ref compressed);
        harness.MarkGpuWritten(compressedId);

        var alias = BlockAliasArray(address, BlockArrayLayers);
        var aliasId = harness.Find(ref alias);

        Assert.False(harness.Image(aliasId).Description.IsBlock);
        Assert.Equal(new[] { aliasId }, harness.Images.FindImagesInRangeForTest(address, BlockImageBytes * BlockArrayLayers, pageOverlap: false));
        Assert.True(harness.Image(aliasId).IsGpuModified);
    }

    [Fact]
    public void FindImage_AnUncompressedViewOfOneLayerOfACompressedImageViewsIt()
    {
        if (!GatePrerequisites.Ready(_vulkan)) return;
        using var harness = new CacheHarness(_vulkan);
        var address = harness.MapBacked(0x20000, ReadWrite);
        var compressed = CompressedArray(address);
        var compressedId = harness.Acquire(ref compressed);

        var alias = BlockAliasArray(address, 1);
        var aliasId = harness.Find(ref alias);

        Assert.Equal(compressedId, aliasId);
        Assert.True(harness.Image(aliasId).Description.IsBlock);
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
