\ -------------------------------------------------------------------------
\ Medium-scale Production Planning Problem in Algebraic LP Format
\
\ A manufacturing company plans production for 3 products (P1, P2, P3) 
\ across 2 consecutive months (M1, M2).
\
\ Decision Variables:
\   - Produce in M1: prod_P1_M1, prod_P2_M1, prod_P3_M1
\   - Produce in M2: prod_P1_M2, prod_P2_M2, prod_P3_M2
\   - Store at end of M1: inv_P1_M1, inv_P2_M1, inv_P3_M1
\
\ Goal: Maximize Profit = Revenue - Production Costs - Storage Costs
\ -------------------------------------------------------------------------

Maximize
 obj: 40 prod_P1_M1 + 50 prod_P2_M1 + 35 prod_P3_M1
    + 42 prod_P1_M2 + 48 prod_P2_M2 + 37 prod_P3_M2
    - 3 inv_P1_M1 - 4 inv_P2_M1 - 2 inv_P3_M1

Subject To
 \ Month 1 Assembly Line Capacity (Machine Hours <= 600)
 m1_mach: 2.0 prod_P1_M1 + 3.0 prod_P2_M1 + 1.5 prod_P3_M1 <= 600.0

 \ Month 1 Skilled Labor Capacity (Labor Hours <= 550)
 m1_labor: 1.5 prod_P1_M1 + 2.0 prod_P2_M1 + 2.5 prod_P3_M1 <= 550.0

 \ Month 2 Assembly Line Capacity (Machine Hours <= 550)
 m2_mach: 2.0 prod_P1_M2 + 3.0 prod_P2_M2 + 1.5 prod_P3_M2 <= 550.0

 \ Month 2 Skilled Labor Capacity (Labor Hours <= 450)
 m2_labor: 1.5 prod_P1_M2 + 2.0 prod_P2_M2 + 2.5 prod_P3_M2 <= 450.0

 \ Minimum Market Demand in Month 1
 dem_P1_M1: prod_P1_M1 - inv_P1_M1 >= 80.0
 dem_P2_M1: prod_P2_M1 - inv_P2_M1 >= 60.0
 dem_P3_M1: prod_P3_M1 - inv_P3_M1 >= 70.0

 \ Minimum Market Demand in Month 2 (includes inventory from Month 1)
 dem_P1_M2: prod_P1_M2 + inv_P1_M1 >= 100.0
 dem_P2_M2: prod_P2_M2 + inv_P2_M1 >= 80.0
 dem_P3_M2: prod_P3_M2 + inv_P3_M1 >= 90.0

 \ Warehouse Storage Capacity at End of Month 1 (sq ft <= 150)
 wh_cap: 1.0 inv_P1_M1 + 1.5 inv_P2_M1 + 0.8 inv_P3_M1 <= 150.0

Bounds
 prod_P1_M1 >= 0
 prod_P2_M1 >= 0
 prod_P3_M1 >= 0
 prod_P1_M2 >= 0
 prod_P2_M2 >= 0
 prod_P3_M2 >= 0
 0 <= inv_P1_M1 <= 60
 0 <= inv_P2_M1 <= 50
 0 <= inv_P3_M1 <= 80

End
