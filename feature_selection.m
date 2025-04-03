function [feature,sorted_indices] = feature_selection(feature,Labels,n,type)
    switch type
        case 1 % Pearson相关系数
            % 计算每个特征与标签的皮尔逊相关系数
            correlations = arrayfun(@(i) corr(feature(i,:)', Labels), 1:size(feature, 1));
            % 对特征按相关系数的绝对值降序排序，并返回排序后的索引
            [~, sorted_indices] = sort(abs(correlations), 'descend');
            feature = feature(sorted_indices(1:n), :);
        case 2 % 卡方验证
            % 需要先将连续特征转换为分类特征，这里采用分位数分箱
            num_bins = 10;
            feature_discrete = zeros(size(feature));
            for i = 1:size(feature, 1)
                feature_discrete(i, :) = discretize(feature(i, :), quantile(feature(i, :), linspace(0, 1, num_bins + 1)));
            end
            % 初始化卡方值数组
            chi2_values = zeros(1, size(feature, 1));
            % 对于每个特征，计算卡方值
            for i = 1:size(feature, 1)
                % 生成列联表
                contingency_table = crosstab(feature_discrete(i, :), Labels);
                
                % 计算卡方值
                chi2_values(i) = compute_chi2(contingency_table);
            end
            % 对特征按卡方值降序排序，并返回排序后的索引
            [~, sorted_indices] = sort(chi2_values, 'descend');
            feature = feature(sorted_indices(1:n), :);
        case 3 % 互信息
            % 初始化互信息值数组
            mi_values = zeros(1, size(feature, 1));
            % 对于每个特征，计算互信息值
            for i = 1:size(feature, 1)
                % 计算互信息值
                mi_values(i) = mutualinfo(feature(i, :)', Labels);
            end
            % 根据互信息值对特征排序
            [~, mi_sorted_indices] = sort(mi_values, 'descend');
            feature = feature(mi_sorted_indices(1:n), :);
    end
end