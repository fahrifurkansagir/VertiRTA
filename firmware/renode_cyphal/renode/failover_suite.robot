# SPDX-License-Identifier: Apache-2.0
# Renode v1.17.0 Headless Dual-STM32H753 OpenCyphal CAN-FD <20 ms Failover Robot Suite.
#
# HONEST CERTIFICATION BOUNDARY (DR-08 Sec 1.5 / 2.3):
#   This suite executes under Renode v1.17.0 (tlib dynamic binary translation) and provides
#   Tier-1 Functional & Protocol SIL verification ONLY — it is NOT hardware WCET evidence.
#   Physical Cortex-M7 execution timing is bounded separately via DWT->CYCCNT on STM32H7 silicon.

*** Settings ***
Suite Setup                   Setup
Suite Teardown                Teardown
Test Setup                    Reset Emulation
Resource                      ${RENODEKEYWORDS}

*** Variables ***
${RESC}                       ${CURDIR}/dual_fcc_failover.resc
${PRIMARY_ELF}                ${CURDIR}/../../../build/renode_cyphal/vertirta_fcc_primary.elf
${SECONDARY_ELF}              ${CURDIR}/../../../build/renode_cyphal/vertirta_fcc_secondary.elf
${FAILOVER_TIMEOUT_S}         0.020
# OpenCyphal v1.0 29-bit extended IDs:
#   Primary FAST SafetyEnvelope (prio=2, subj=1024, node=10)      = 0x0864000A
#   Secondary EXCEPTIONAL handover (prio=0, subj=1024, node=11)   = 0x0064000B
${PRIMARY_ENV_CAN_ID}         0x0864000A
${SECONDARY_HANDOVER_CAN_ID}  0x0064000B
# Malformed 64-byte CAN-FD babbling-idiot payload with corrupted CRC-32C (0xDEADBEEF) & valid tail 0xE1
${CORRUPT_CRC_FRAME}          0000000000000000000000000000000000000000000000000000000000000000010000000000000000000000EFBEADDE000000000000000000000000000000E1

*** Keywords ***
Boot Dual FCC Lockstep Pair
    Execute Command           $primary_elf=@${PRIMARY_ELF}
    Execute Command           $secondary_elf=@${SECONDARY_ELF}
    Execute Script            ${RESC}
    ${pri_uart}=              Create Terminal Tester    sysbus.usart3    machine=fcc_primary      defaultPauseEmulation=true
    ${sec_uart}=              Create Terminal Tester    sysbus.usart3    machine=fcc_secondary    defaultPauseEmulation=true
    ${can_tester}=            Create CAN Tester         canHub           0.020
    Start Emulation
    Wait For Line On Uart     BOOT_OK ROLE=PRIMARY NODE_ID=10     testerId=${pri_uart}    timeout=1.0
    Wait For Line On Uart     BOOT_OK ROLE=SECONDARY NODE_ID=11   testerId=${sec_uart}    timeout=1.0
    # Confirm nominal 400 Hz Primary SafetyEnvelope traffic on Subject-ID 1024 before fault injection.
    Wait For Frame With Id    ${PRIMARY_ENV_CAN_ID}    timeout=0.020    testerId=${can_tester}    pauseEmulation=true
    RETURN                    ${pri_uart}    ${sec_uart}    ${can_tester}

Inject Corrupted Babbling Idiot Frame
    Execute Command           py "from Antmicro.Renode.Core.CAN import CANMessageFrame; from System import Array, Byte; raw = bytearray.fromhex('${CORRUPT_CRC_FRAME}'); f = CANMessageFrame.CreateWithExtendedId(${PRIMARY_ENV_CAN_ID}, Array[Byte](raw), True, False, True, True); emulationManager.Instance.CurrentEmulation['fcc_secondary']['sysbus.fdcan1'].OnFrameReceived(f)"

*** Test Cases ***
Fault 1 Primary HardFault Lockup Triggers Secondary Takeover Under 20ms Virtual Time
    [Documentation]           Injects an architectural Cortex-M7 HardFault into Primary NVIC via
    ...                       SetPendingSynchronousFault and asserts Secondary Node 11 takes
    ...                       bus authority on Subject-ID 1024 within timeout=0.020 (Functional & Protocol SIL ONLY).
    ${pri_uart}  ${sec_uart}  ${can_tester}=    Boot Dual FCC Lockstep Pair
    Execute Command           mach set "fcc_primary"
    Execute Command           sysbus.nvic SetPendingSynchronousFault 3
    Wait For Line On Uart     FAILOVER_ACTIVE    testerId=${sec_uart}    timeout=0.020
    Wait For Frame With Id    ${SECONDARY_HANDOVER_CAN_ID}    timeout=0.020    testerId=${can_tester}    pauseEmulation=true

Fault 2 Primary CAN Bus Disconnect Triggers Secondary Takeover Under 20ms Virtual Time
    [Documentation]           Disconnects Primary FDCAN1 from virtual CANHub (modeling wiring/transceiver
    ...                       open fault) and asserts Secondary Node 11 takeover within timeout=0.020.
    ${pri_uart}  ${sec_uart}  ${can_tester}=    Boot Dual FCC Lockstep Pair
    Execute Command           mach set "fcc_primary"
    Execute Command           connector Disconnect sysbus.fdcan1 canHub
    Wait For Line On Uart     FAILOVER_ACTIVE    testerId=${sec_uart}    timeout=0.020
    Wait For Frame With Id    ${SECONDARY_HANDOVER_CAN_ID}    timeout=0.020    testerId=${can_tester}    pauseEmulation=true

Fault 3 Corrupted CRC Babbling Idiot Rejected And Triggers Secondary Takeover Under 20ms
    [Documentation]           Silences healthy Primary and injects 400 Hz corrupted-CRC frames on Subject-ID 1024
    ...                       into Secondary FDCAN1; Secondary must reject invalid CRC-32C without refreshing watchdog
    ...                       and transition to FAILOVER_ACTIVE within timeout=0.020.
    ${pri_uart}  ${sec_uart}  ${can_tester}=    Boot Dual FCC Lockstep Pair
    Execute Command           mach set "fcc_primary"
    Execute Command           connector Disconnect sysbus.fdcan1 canHub
    FOR    ${i}    IN RANGE    4
        Inject Corrupted Babbling Idiot Frame
        Execute Command       emulation RunFor "0.0025"
    END
    Wait For Line On Uart     FAILOVER_ACTIVE.*CRC_REJECTS=[1-9]    testerId=${sec_uart}    timeout=0.020    treatAsRegex=true
    Wait For Frame With Id    ${SECONDARY_HANDOVER_CAN_ID}    timeout=0.020    testerId=${can_tester}    pauseEmulation=true
