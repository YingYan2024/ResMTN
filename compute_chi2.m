function chi2_value = compute_chi2(contingency_table)
    % 计算期望频数
    row_sums = sum(contingency_table, 2);
    col_sums = sum(contingency_table, 1);
    total_sum = sum(row_sums);
    expected_table = row_sums * col_sums / total_sum;
    
    % 计算卡方值
    chi2_value = sum(sum((contingency_table - expected_table).^2 ./ expected_table));
end