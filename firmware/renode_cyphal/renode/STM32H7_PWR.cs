// SPDX-License-Identifier: Apache-2.0
// Custom Renode C# Peripheral for STM32H753 / H743 PWR Controller (0x58024800).
// Resolves upstream Renode issues #649 and #857 where PWR_D3CR (offset 0x18, VOSRDY)
// and supply-readiness bits return 0 and hang unmodified bare-metal VOS0/LDO boot loops.

using Antmicro.Renode.Core;
using Antmicro.Renode.Core.Structure.Registers;
using Antmicro.Renode.Peripherals.Bus;

namespace Antmicro.Renode.Peripherals.Miscellaneous
{
    public class STM32H7_PWR : BasicDoubleWordPeripheral, IKnownSize
    {
        public STM32H7_PWR(IMachine machine) : base(machine)
        {
            DefineRegisters();
            Reset();
        }

        public long Size => 0x400;

        public override void Reset()
        {
            base.Reset();
            cr1 = 0xF000C000u;
            cr2 = 0x00000000u;
            cr3 = 0x00000006u;  // LDOEN=1, SCUEN=1
            cpucr = 0x00000000u;
            d3cr = 0x00004000u; // VOS3 default; VOSRDY (bit 13) asserted on read
        }

        private void DefineRegisters()
        {
            Registers.CR1.Define(this)
                .WithValueField(0, 32,
                    valueProviderCallback: _ => cr1,
                    writeCallback: (_, v) => cr1 = (uint)v);

            // CSR1 (0x04): ACTVOSRDY (bit 13) always set; ACTVOS[15:14] tracks D3CR.VOS[15:14].
            Registers.CSR1.Define(this)
                .WithValueField(0, 32, FieldMode.Read,
                    valueProviderCallback: _ => (d3cr & 0x0000C000u) | (1u << 13));

            Registers.CR2.Define(this)
                .WithValueField(0, 32,
                    valueProviderCallback: _ => cr2,
                    writeCallback: (_, v) => cr2 = (uint)v);

            // CR3 (0x0C): Assert USB33RDY (bit 26) immediately when USB33DEN (bit 24) is enabled.
            Registers.CR3.Define(this)
                .WithValueField(0, 32,
                    valueProviderCallback: _ =>
                    {
                        var val = cr3;
                        if ((val & (1u << 24)) != 0u)
                        {
                            val |= (1u << 26);
                        }
                        return val;
                    },
                    writeCallback: (_, v) => cr3 = (uint)v);

            Registers.CPUCR.Define(this)
                .WithValueField(0, 32,
                    valueProviderCallback: _ => cpucr,
                    writeCallback: (_, v) => cpucr = (uint)v);

            // D3CR (0x18): Assert VOSRDY (bit 13) so VOS1/VOS0 clock transitions never deadlock.
            Registers.D3CR.Define(this)
                .WithValueField(0, 32,
                    valueProviderCallback: _ => d3cr | (1u << 13),
                    writeCallback: (_, v) => d3cr = (uint)v);
        }

        private uint cr1;
        private uint cr2;
        private uint cr3;
        private uint cpucr;
        private uint d3cr;

        private enum Registers : long
        {
            CR1 = 0x00,
            CSR1 = 0x04,
            CR2 = 0x08,
            CR3 = 0x0C,
            CPUCR = 0x10,
            D3CR = 0x18,
        }
    }
}
