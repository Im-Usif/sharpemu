// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Vulkan;

public static partial class Gen5SpirvTranslator
{
    // The argument a test takes for a callee whose every record matches (ShaderCallMatchControl.RuntimeArgument): no guest
    // address is this large.
    private const ulong AnyCallArgument = ulong.MaxValue;

    private sealed partial class CompilationContext
    {
        // The slot of each nonzero record argument among the call arguments, when the host pushes the arguments.
        private readonly Dictionary<ulong, int>? _callArgumentSlots;

        // The slot of each callee's address among the call targets, when the host pushes the targets.
        private readonly Dictionary<ulong, int>? _callTargetSlots;

        // A callee's address: the host's call target for it, or the link-time address without pushed targets.
        private uint CallTarget(ulong address)
        {
            if (_callTargetSlots is null || !_callTargetSlots.TryGetValue(address, out var slot))
            {
                return ULong(address);
            }

            return LoadCallData(_request.Bindings.CallTargetsDword + (uint)slot * 2);
        }

        // A pair of call data dwords. The layout sizes the shader data for them, so the load needs no bounds check.
        private uint LoadCallData(uint dword)
        {
            var (low, high) = LoadCallDataWords(dword);
            return Pair64(low, high);
        }

        private (uint Low, uint High) LoadCallDataWords(uint dword)
        {
            uint Word(uint index) => _request.Bindings.UsesPushData
                ? LoadShaderDataDword(UInt(index))
                : Load(_uintType, BlockWordPointer(_shaderData, UInt(index)));
            return (Word(dword), Word(dword + 1));
        }

        // A record's argument is compared with the host's argument for its slot, so the module does not depend on where
        // the game placed its argument data; a null argument stays the constant. The argument register pair is compared
        // a dword at a time, and each argument is loaded once in the current block of code.
        private uint CallArgumentMatches((uint Low, uint High) argument, ulong recordArgument, Dictionary<int, (uint Low, uint High)> loaded)
        {
            if (recordArgument == AnyCallArgument)
            {
                return _module.ConstantBool(true);
            }

            (uint Low, uint High) value;
            if (recordArgument == 0 || _callArgumentSlots is null || !_callArgumentSlots.TryGetValue(recordArgument, out var slot))
            {
                value = (UInt((uint)recordArgument), UInt((uint)(recordArgument >> 32)));
            }
            else if (!loaded.TryGetValue(slot, out value))
            {
                value = LoadCallDataWords(_request.Bindings.CallArgumentsDword + (uint)slot * 2);
                loaded.Add(slot, value);
            }

            return LogicalAnd(
                _module.AddInstruction(SpirvOp.IEqual, _boolType, argument.Low, value.Low),
                _module.AddInstruction(SpirvOp.IEqual, _boolType, argument.High, value.High));
        }
    }
}
