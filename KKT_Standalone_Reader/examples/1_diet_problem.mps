NAME          DIET_PROBLEM
* -------------------------------------------------------------------------
* Medium-scale Diet Optimization Problem in Standard MPS Format
*
* Goal: Find the lowest-cost weekly meal plan that satisfies all nutritional
*       requirements without exceeding maximum dietary limits.
*
* 8 Foods (Variables):
*   OATM (Oatmeal), CHIK (Chicken), EGGS (Eggs), MILK (Milk),
*   APPL (Apples), BRED (Bread), RICE (Rice), FISH (Fish)
*
* 5 Nutritional Requirements (Constraints):
*   CAL  (Calories,     >= 14000 kcal)
*   PROT (Protein,      >= 400 g)
*   FAT  (Fat,          <= 350 g)
*   CARB (Carbs,        >= 1500 g)
*   VITA (Vitamin A,    >= 5000 IU)
* -------------------------------------------------------------------------
ROWS
 N  COST
 G  CAL
 G  PROT
 L  FAT
 G  CARB
 G  VITA
COLUMNS
    OATM      COST             1.50   CAL            350.0
    OATM      PROT             12.0   FAT              6.0
    OATM      CARB             60.0   VITA             0.0
    CHIK      COST             5.00   CAL            250.0
    CHIK      PROT             30.0   FAT              4.0
    CHIK      CARB              0.0   VITA           100.0
    EGGS      COST             2.00   CAL            150.0
    EGGS      PROT             12.0   FAT             10.0
    EGGS      CARB              1.0   VITA           500.0
    MILK      COST             2.50   CAL            120.0
    MILK      PROT              8.0   FAT              5.0
    MILK      CARB             12.0   VITA           400.0
    APPL      COST             1.00   CAL             90.0
    APPL      PROT              0.5   FAT              0.3
    APPL      CARB             25.0   VITA           200.0
    BRED      COST             1.20   CAL            200.0
    BRED      PROT              6.0   FAT              1.5
    BRED      CARB             40.0   VITA             0.0
    RICE      COST             1.80   CAL            300.0
    RICE      PROT              5.0   FAT              0.5
    RICE      CARB             65.0   VITA             0.0
    FISH      COST             6.50   CAL            200.0
    FISH      PROT             26.0   FAT              7.0
    FISH      CARB              0.0   VITA           300.0
RHS
    RHS1      CAL           14000.0   PROT           400.0
    RHS1      FAT             350.0   CARB          1500.0
    RHS1      VITA           5000.0
BOUNDS
 UP BND1      OATM             20.0
 UP BND1      CHIK             15.0
 UP BND1      EGGS             25.0
 UP BND1      MILK             20.0
 UP BND1      APPL             30.0
 UP BND1      BRED             20.0
 UP BND1      RICE             20.0
 UP BND1      FISH             10.0
ENDATA
