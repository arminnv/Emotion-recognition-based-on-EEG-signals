fs = 1000;
X = [];

for i=1:size(TrainData, 3)
    disp(i)
    X = [X; extract_features(TrainData(:, :, i), fs)];
end
disp("finnished")
%%
Xtst = [];
for i=1:size(TestData, 3)
    disp(i)
    Xtst = [Xtst; extract_features(TestData(:, :, i), fs)];
end
disp("finnished")
%%

Y = TrainLabels';

n_folds = 5;
n_data = size(X, 1);
n_selected = 10:10:300;

accuracy_test = zeros(length(n_selected), 1);
accuracy_train = zeros(length(n_selected), 1);

Names = get_index_names(59);
for k=1:n_folds
    idx_test = 1+(k-1)*round(n_data/n_folds):k*(round(n_data/n_folds));
    idx_train = 1:n_data;
    idx_train(idx_test) = [];
    
    %selected_features = selected_features(:, IDX(1:280));
    Index_names = get_index_names(59);
    X_normal = normalize(X, 1);
    i_nan = any(isnan(X_normal),1);
    X_normal(:, i_nan) = [];
    Index_names(i_nan) = [];

    X_train = X_normal(idx_train, :);
    Y_train = Y(idx_train);
    
    mu0 = mean(X_train, 1);
    
    n_1 = sum(Y_train == 1);
    mu_1 = mean(X_train(find(Y_train == 1), :), 1);
    var_1 = var(X_train(find(Y_train == 1), :), 1);
    
    n_2 = sum(Y_train == -1);
    mu_2 = mean(X_train(find(Y_train == -1), :), 1);
    var_2 = var(X_train(find(Y_train == -1), :), 1);
    
    inter_class = n_1*(mu_1-mu0).^2 + n_2*(mu_2-mu0).^2; 
    intra_class = (n_1-1)*var_1 + (n_2-1)*var_2;
    
    score = inter_class ./ intra_class;
    
    [B, Index] = sort(score, 'descend');
    score = score(Index);
    score_names = Index_names(Index);
    
    %Mdl = fitctree(X_train, Y_train,'SplitCriterion','gdi');
    %imp = predictorImportance(Mdl);
    
    for n=1:length(n_selected)  
        X_train = X_normal(idx_train, Index(1:n_selected(n)));
        Y_train = Y(idx_train);
        X_test = X_normal(idx_test, Index(1:n_selected(n)));
        Y_test = Y(idx_test);
        
        %Mdl = fitcsvm(X_train, Y_train,'KernelFunction','rbf', 'KernelScale',70);
        %Mdl = fitcknn(X_train, Y_train, "NumNeighbors", 5);
        %Mdl = fitctree(X_train, Y_train,'SplitCriterion','gdi');
        Mdl = fitctree(X_train, Y_train);
        imp = predictorImportance(Mdl); 
        %Mdl = fitcensemble(X_train, Y_train,'Method','AdaBoostM1','NumLearningCycles',50,'Learners','tree');
        Ypred = predict(Mdl, X_test);
        accuracy_test(n) = accuracy_test(n) + mean(Ypred==Y_test)/n_folds;
        accuracy_train(n) = accuracy_train(n) + mean(predict(Mdl, X_train)==Y_train)/n_folds;
    end
end
[accuracy_best, i_best] = max(accuracy_test);

fprintf("test: %d , best n = %d \n", accuracy_best, n_selected(i_best))
fprintf("train: %d \n", accuracy_train(i_best))



function features = extract_features(X, fs)
    X = bandpass(X', [0.1, 100], fs)';
    
    n_channels = size(X, 1);
    L = size(X, 2);
    
    % Frequency bands: delta, theta, alpha, low-beta, mid-beta, high-beta, gamma
    f_bands = [0.1, 4, 8, 12, 16, 20, 30, 100];
    
    features = [];
    
    f = (0:L/2-1)*fs/L;
    
    % Spectral Energy
    SE = zeros(n_channels, length(f_bands)-1);

    for ch=1:n_channels
        signal = X(ch, :);
        % Power Spectral Density
        [Pxx, f] = pwelch(signal, [], [], [], fs);
        
        % Frequency of highest peak
        f_peak = f(find(Pxx==max(Pxx)));

        % Average frequency
        f_avg = sum(Pxx.*f)/sum(Pxx);
        
        % Median frequency
        f_med = medfreq(Pxx, f);
        
        features = [features, f_peak, f_avg, f_med];
    end
    
    for b=1:length(f_bands)-1
        % Bandpass filtering signal
        f_low = f_bands(b);
        f_high = f_bands(b+1);        
        X_filtered = bandpass(X', [f_low, f_high], fs)';
    
        for ch=1:n_channels
            signal = X_filtered(ch, :);
        
            % Variance of signal
            Var = var(signal);
        
            % Histogram counts
            [N, edges] = histcounts(signal, 10);
        
            % AR coefficients
            [a, e] = lpc(signal, 10);
        
            % Form Factor
            dif1 = diff(signal, 1);
            dif2 = diff(signal, 2);
            FF = (var(dif2)/var(dif1)) / (var(dif1)/var(signal));
    
            % Power Spectral Density
            [Pxx, f] = pwelch(signal, [], [], [], fs);
            
            % Frequency of highest peak
            f_peak = f(find(Pxx==max(Pxx)));
    
            % Average frequency
            f_avg = sum(Pxx.*f)/sum(Pxx);
            
            % Median frequency
            f_med = medfreq(Pxx, f);
            
            % Spectral energy of channel ch and band b
            SE(ch, b) = sum(Pxx);
            
            features = [features, Var, N, a, FF, f_peak, f_avg, f_med];
        end
        
        % Correlation between channels
        A = corrcoef(X');
        R = reshape(tril(A, -1), 1, []);
        R(R==0) = [];
        features = [features, R];
        
        features = [features, efficiency_wei(A), betweenness_wei(A)', diffusion_efficiency(A)];

        % Graph frequencies
        L = diag(sum(A, 1)) - A;
        
        Eigen_values = eig(L);
    
        features = [features, Eigen_values'];
    end
    
    for ch=1:n_channels
        sum_energy = sum(SE(ch, :));
        for b=1:length(f_bands)-1
            % Power Spectral Ration
            PSR = SE(ch, b)/sum_energy;
            features = [features, PSR];
        end
    end

    %for i=1:n_channels-1
    %    for j=i+1:n_channels
            % Cross Power Spectral Density
    %        [pxy, f] = cpsd(X(i, :), X(j, :), [], [], [], fs);
    %        features = [features, mean(pxy)];
    %    end
    %end
end


function Index_names = get_index_names(n_channels)    
  
    % Frequency bands: delta, theta, alpha, low-beta, mid-beta, high-beta, gamma
    f_bands = ["delta", "theta", "alpha", "low-beta", "mid-beta", "high-beta", "gamma"];
    
    Index_names = [];

    for ch=1:n_channels
        CH = num2str(ch);
        Index_names = [Index_names, "fpeak"+CH, "favg"+CH, "fmed"+CH];
    end

    for b=1:length(f_bands)
    
        for ch=1:n_channels
            CH = num2str(ch)+f_bands(b);
            Index_names = [Index_names, "Var"+CH, repelem("N"+CH, 10), repelem("a"+CH, 11), "FF"+CH, "fpeak"+CH, "favg"+CH, "fmed"+CH];
        end
        
        % Correlation between channels
        Index_names = [Index_names, repelem("R"+f_bands(b), 1711)];

        Index_names = [Index_names, "NE"+f_bands(b), repelem("BC"+f_bands(b), 59), "DE"+f_bands(b)];

        % Graph frequencies
        Index_names = [Index_names, repelem("Eig"+f_bands(b), 59)];
    end
    
    for ch=1:n_channels
        for b=1:length(f_bands)
            % Power Spectral Ration
            Index_names = [Index_names, "PSR"+num2str(ch)+f_bands(b)];
        end
    end
end

