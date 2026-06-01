# Battery State of Health and Degradation

## State of Health (SOH)

State of Health (SOH) expresses a battery's current maximum capacity as a percentage of its
original rated capacity. A cell at 100% SOH delivers its full nameplate capacity; a cell at
80% SOH delivers only 80%. For grid energy-storage systems, **80% SOH is the conventional
end-of-life threshold** — beyond this point the usable energy and round-trip efficiency have
degraded enough that the asset is typically retired or repurposed.

SOH is distinct from State of Charge (SOC). SOC is how full the battery is right now (0–100%
of *current* capacity), whereas SOH is how much total capacity the battery has lost
permanently over its life.

## Degradation mechanisms

Lithium-ion degradation is driven by two broad families of mechanism:

- **Calendar ageing** — capacity loss that occurs simply with the passage of time, even when
  the cell is idle. It is accelerated by high State of Charge and high temperature. Growth of
  the solid-electrolyte interphase (SEI) layer is the dominant calendar-ageing mechanism.
- **Cycle ageing** — capacity loss caused by charge/discharge cycling. Each full cycle
  consumes a small amount of active lithium and mechanically stresses the electrodes. Lithium
  plating during fast charging at low temperature is a key cycle-ageing mechanism.

## Factors that accelerate degradation

The main stressors that shorten battery life are:

1. **High temperature.** Degradation roughly doubles for every 10 °C increase. Keeping cells
   between 15 °C and 35 °C materially extends life.
2. **High depth of discharge (DoD).** Cycling a cell over a wider SOC window (e.g. 0–100%)
   ages it faster than shallow cycling (e.g. 30–70%).
3. **High C-rate.** Charging or discharging at high current relative to capacity increases
   heat generation and mechanical stress.
4. **Sustained high State of Charge.** Storing cells near 100% SOC accelerates calendar
   ageing; ~50% SOC is gentler for long-term storage.
