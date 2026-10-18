// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers;
using System.IO.Hashing;
using SharpEmu.HLE;

namespace SharpEmu.Libs.Gpu.Images;

// The image readbacks of one cache that have not landed yet, in the order they were queued.
internal sealed class ReadbackLandings
{
    private readonly object _gate = new();
    private ReadbackLanding[] _pending = [];

    public ReadbackLanding[] Pending => Volatile.Read(ref _pending);

    // A readback queued over these guest bytes.
    public ReadbackLanding Queue(ulong address, ReadOnlySpan<byte> snapshot)
    {
        var landing = new ReadbackLanding(this, address, snapshot);
        lock (_gate)
        {
            _pending = [.. _pending, landing];
        }

        return landing;
    }

    public void Remove(ReadbackLanding landing)
    {
        lock (_gate)
        {
            var index = Array.IndexOf(_pending, landing);
            if (index >= 0)
            {
                _pending = [.. _pending.AsSpan(0, index), .. _pending.AsSpan(index + 1)];
            }
        }
    }
}

// A readback lands in guest memory when its tick completes. By then the guest may have stored
// into the same bytes, in memory a title freed and gave to its own data. On the console the
// GPU's bytes reached memory first and the store replaced them, so a page whose bytes changed
// since the readback was queued keeps the guest's bytes; the other pages take the readback.
internal sealed class ReadbackLanding
{
    private const int PageBytes = 4096;
    private const int WindowBytes = 1 << 20;
    private static int _reports;

    private readonly ReadbackLandings _queue;
    private readonly ulong _address;
    private readonly int _size;
    private readonly ulong[] _pageHashes;

    internal ReadbackLanding(ReadbackLandings queue, ulong address, ReadOnlySpan<byte> snapshot)
    {
        _queue = queue;
        _address = address;
        _size = snapshot.Length;
        _pageHashes = new ulong[(snapshot.Length + PageBytes - 1) / PageBytes];
        for (var page = 0; page < _pageHashes.Length; page++)
        {
            _pageHashes[page] = HashPage(snapshot, page * PageBytes);
        }
    }

    // Writes the readback over every page the guest has not stored to since it was queued.
    // Landings run one at a time.
    public bool TryLand(IGuestBackedSpace backing, ReadOnlySpan<byte> readback)
    {
        var kept = 0;
        var window = ArrayPool<byte>.Shared.Rent(Math.Min(readback.Length, WindowBytes));
        Span<bool> lands = stackalloc bool[WindowBytes / PageBytes];
        try
        {
            lock (_queue)
            {
                for (var windowStart = 0; windowStart < readback.Length; windowStart += WindowBytes)
                {
                    var current = window.AsSpan(0, Math.Min(WindowBytes, readback.Length - windowStart));
                    if (!backing.TryReadBacking(_address + (ulong)windowStart, current))
                    {
                        return false;
                    }

                    var pages = (current.Length + PageBytes - 1) / PageBytes;
                    for (var index = 0; index < pages; index++)
                    {
                        lands[index] = HashPage(current, index * PageBytes) == _pageHashes[windowStart / PageBytes + index];
                        kept += lands[index] ? 0 : 1;
                    }

                    // Unchanged pages land in runs, one write per run.
                    for (var first = 0; first < pages;)
                    {
                        if (!lands[first])
                        {
                            first++;
                            continue;
                        }

                        var end = first + 1;
                        while (end < pages && lands[end])
                        {
                            end++;
                        }

                        var start = windowStart + first * PageBytes;
                        var length = Math.Min(end * PageBytes, current.Length) - first * PageBytes;
                        if (!TryWriteRun(backing, start, readback.Slice(start, length)))
                        {
                            return false;
                        }

                        first = end;
                    }
                }
            }
        }
        finally
        {
            _queue.Remove(this);
            ArrayPool<byte>.Shared.Return(window);
        }

        if (kept != 0 && Interlocked.Increment(ref _reports) <= 64)
        {
            Console.Error.WriteLine($"[GPU][INFO] readback 0x{_address:X16}+0x{readback.Length:X}: {kept} page(s) the guest stored to since it was queued keep the guest's bytes");
        }

        return true;
    }

    // What a landing writes is no guest store to the readbacks still pending over those bytes:
    // a page they still hold unchanged takes the new bytes as its base, so the readback that
    // lands last still writes last.
    private bool TryWriteRun(IGuestBackedSpace backing, int start, ReadOnlySpan<byte> bytes)
    {
        var address = _address + (ulong)start;
        List<(ReadbackLanding Other, int Page)>? followers = null;
        foreach (var other in _queue.Pending)
        {
            if (other != this)
            {
                other.CollectUnchangedPages(backing, address, (ulong)bytes.Length, ref followers);
            }
        }

        if (!backing.TryWriteBacking(address, bytes))
        {
            return false;
        }

        if (followers != null)
        {
            foreach (var (other, page) in followers)
            {
                other.RehashPage(backing, page);
            }
        }

        return true;
    }

    private void CollectUnchangedPages(IGuestBackedSpace backing, ulong address, ulong size, ref List<(ReadbackLanding, int)>? pages)
    {
        var begin = Math.Max(address, _address);
        var end = Math.Min(address + size, _address + (ulong)_size);
        for (var page = begin < end ? (int)((begin - _address) / PageBytes) : _pageHashes.Length;
             page < _pageHashes.Length && _address + (ulong)page * PageBytes < end; page++)
        {
            if (TryHashGuestPage(backing, page, out var hash) && hash == _pageHashes[page])
            {
                (pages ??= []).Add((this, page));
            }
        }
    }

    private void RehashPage(IGuestBackedSpace backing, int page)
    {
        if (TryHashGuestPage(backing, page, out var hash))
        {
            _pageHashes[page] = hash;
        }
    }

    private bool TryHashGuestPage(IGuestBackedSpace backing, int page, out ulong hash)
    {
        Span<byte> bytes = stackalloc byte[PageBytes];
        var length = Math.Min(PageBytes, _size - page * PageBytes);
        hash = 0;
        if (!backing.TryReadBacking(_address + (ulong)page * PageBytes, bytes[..length]))
        {
            return false;
        }

        hash = XxHash3.HashToUInt64(bytes[..length]);
        return true;
    }

    private static ulong HashPage(ReadOnlySpan<byte> bytes, int start) =>
        XxHash3.HashToUInt64(bytes.Slice(start, Math.Min(PageBytes, bytes.Length - start)));
}
