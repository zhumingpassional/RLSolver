\ Model restricted_master_problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  20 theta[0] + 96 theta[1] + 60 theta[2]
Subject To
 a_1: theta[2] >= 1
 a_2: theta[2] >= 1
 a_3: theta[2] >= 1
 a_4: theta[2] >= 1
 a_5: theta[2] >= 1
 a_6: theta[2] >= 1
 a_7: theta[2] >= 1
 a_8: theta[2] >= 1
 a_9: theta[2] >= 1
 a_10: theta[2] >= 1
 a_11: theta[2] >= 1
 a_12: theta[1] >= 1
 a_13: theta[1] >= 1
 a_14: theta[1] >= 1
 a_15: theta[1] >= 1
 a_16: theta[1] >= 1
 a_17: theta[1] >= 1
 a_18: theta[1] >= 1
 a_19: theta[1] >= 1
 a_20: theta[0] >= 1
 b: theta[0] + theta[1] + theta[2] <= 25
Bounds
Generals
 theta[0] theta[1] theta[2]
End
