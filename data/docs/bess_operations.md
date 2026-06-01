# BESS Operating Parameters and Safety

## Key operating parameters

- **C-rate** is the charge/discharge current normalised to capacity. A 1C rate fully charges
  or discharges the battery in one hour; 0.5C takes two hours; 2C takes thirty minutes.
- **Depth of Discharge (DoD)** is the fraction of capacity withdrawn in a cycle. Lower DoD
  generally yields more cycles before end of life.
- **Round-trip efficiency** is the ratio of energy out to energy in over a full
  charge/discharge cycle. Modern lithium-ion BESS achieve roughly 85–95% round-trip
  efficiency; losses appear as heat.

## Thermal management

Thermal management keeps cells within their safe and efficient operating window. Liquid
cooling is common in large containerised systems because it removes heat more effectively
than air cooling at high power. The goal is to limit both the absolute cell temperature and
the temperature *gradient* across the pack, since uneven temperatures cause cells to age at
different rates and drift out of balance.

## Thermal runaway

**Thermal runaway** is a self-sustaining exothermic reaction in which a cell generates heat
faster than it can dissipate, driving neighbouring cells into the same condition and
potentially causing fire. Common triggers are internal short circuits, mechanical damage,
overcharging, and external heating. Mitigations include cell-level fusing, physical spacing
and barriers between cells, gas detection, and deflagration venting in the enclosure.

## Battery Management System (BMS)

The Battery Management System (BMS) monitors cell voltages, currents, and temperatures, and
enforces safe operating limits. Its core responsibilities are:

- **Cell balancing** — equalising the charge across series-connected cells so no single cell
  is over- or under-charged.
- **Protection** — opening contactors on over-voltage, under-voltage, over-current, or
  over-temperature events.
- **State estimation** — computing State of Charge (SOC) and State of Health (SOH) from
  measured signals.
