close all



% Number of selected features
n_select = [1000];
accuracy_test = zeros(length(n_select), 1);
accuracy_train = zeros(length(n_select), 1);

n_folds = 5;
for m=1:length(n_select)
    for k=1:n_folds
        idx_test = 1+(k-1)*round(n_data/n_folds):k*(round(n_data/n_folds));
        idx_train = 1:n_data;
        idx_train(idx_test) = [];
        
        %selected_features = selected_features(:, IDX(1:280));
        Index_names = get_index_names(59);
        X_normal = normalize((X), 1);
        %X_normal(:, find(Index==0))=0;
        i_nan = any(isnan(X_normal),1);
        X_normal(:, i_nan) = [];
        Index_names(i_nan) = [];

        % Number of features
        n_features = size(X_normal, 2);
    
        X_train = X_normal(idx_train, :);
        Y_train = Y(idx_train);
        
        popsize = 200; % Population size
        % Set nondefault solver options
        options = optimoptions("ga","MaxGenerations", 200, "CreationFcn","gacreationuniformint",...
             "InitialPopulationMatrix", pop_create(popsize, n_select(m), n_features),"CrossoverFcn",...
             "crossoverscattered", "PopulationSize", popsize,'MutationFcn',{@mutationuniform,0.1}, ...
            "PlotFcn",["gaplotbestf"]);
    
        % Solve
        intcon = 1:n_select(m);
        lb = ones([n_select(m), 1]);
        %hb = ones([n_features, 1]);
        [solution,objectiveValue] = ga({@fittness_fisher, X_train, Y_train},n_select(m),[],[],[],[],lb,lb*n_features,[],intcon,options);
        solution = unique(solution);
        score_names = Index_names(solution);
        % Clear variables
        clearvars options
        
        X_reduced = X_normal(:, solution);
        
        Xtrain = X_reduced(idx_train, :);
        Xtest = X_reduced(idx_test, :);
        Ytrain = Y(idx_train);
        Ytest = Y(idx_test);
        
        %Mdl = fitcsvm(X_train, Y_train,'KernelFunction','rbf', 'KernelScale',70);
        %Mdl = fitcknn(X_train, Y_train, "NumNeighbors", 6);
        Mdl = fitctree(Xtrain, Ytrain);
        Ypred = predict(Mdl, Xtest);
        accuracy_test(m) = accuracy_test(m) + mean(Ypred==Ytest)/n_folds;
        accuracy_train(m) = accuracy_train(m) + mean(predict(Mdl, Xtrain)==Ytrain)/n_folds;
    end
    fprintf("test: %d n_select = %d \n", accuracy_test(m), n_select(m))
end

[accuracy_best, i_best] = max(accuracy_test);

fprintf("test: %d , best n = %d \n", accuracy_best, n_select(i_best))
fprintf("train: %d \n", accuracy_train(i_best))



function J = fittness_fisher(x, X_train, Y_train)

    x = unique(x);

    X = X_train(:, x);
    Y = Y_train;

    mu0 = mean(X, 1);
    
    idx1 = find(Y==1);
    n1 = sum(Y == 1, 1);
    mu1 = mean(X(idx1, :));
    S1 = zeros(size(X, 2));
    for i=idx1
        S1 = S1 + (X(i, :)-mu1)'*(X(i, :)-mu1) / n1;
    end
    
    idx2 = find(Y==-1);
    n2 = sum(Y == -1, 1);
    mu2 = mean(X(idx2, :));
    S2 = zeros(size(X, 2));
    for i=idx2
        S2 = S2 + (X(i, :)-mu2)'*(X(i, :)-mu2) / n2;
    end

    Sw = (n1*S1 + n2*S2)/(n1 + n2);
    Sb = (mu1 - mu0)'*(mu1 - mu0) + (mu2 - mu0)'*(mu2 - mu0);

    J = -trace(Sb)/trace(Sw);
    %J = -(mu1 - mu2) * inv(Sw) * (mu1 - mu2)';
end

function pop = initpop(popsize, nvars, nbits)
    % Initialize population with binary vectors of length nvars
    % Each vector has exactly nbits ones
    pop = zeros(popsize, nvars); % Preallocate population matrix
    for i = 1:popsize
        idx = randperm(nvars, nbits); % Randomly select nbits positions
        pop(i, idx) = 1; % Set those positions to 1
    end
end


function Kids = fisher_crossover(parents,options,nvars,FitnessFcn,thisScore,thisPopulation)

    M = length(parents)/2  ;
    Kids = zeros(M,nvars) ;
    
    for i=1:M
        p1 = thisPopulation(parents(2*i-1),:) ;
        p2 = thisPopulation(parents(2*i),:) ;

        idx = [find(p1==1), find(p2==1)];
        idx = idx(randperm(length(idx)));
        
        Kid = zeros(length(p1), 1);
        Kid(idx) = 1;
        
        Kids(i,:) = Kid';
    end
    
end

function Kids = permute_mutation(parents,options,nvars,FitnessFcn, state, thisScore,thisPopulation)
    
    Kids = zeros(length(parents),nvars) ;

    for m=1:length(parents)
        parent = thisPopulation(parents(m),:);
        n = length(parent);
        newKid = parent;
        for k=1:3
            % randomly select two indices to swap
            i = randi(n);
            j = randi(n);
            % create a copy of the vector and swap the elements
            newKid([i j]) = newKid([j i]);
        end
        Kids(m,:) = newKid ;
    end
end

function pop = pop_create(popsize, nvars, max_index)
    % Initialize population with binary vectors of length nvars
    % Each vector has exactly nbits ones
    pop = zeros(popsize, nvars); % Preallocate population matrix
    for i = 1:popsize
        idx = randi(max_index, [1, nvars]);
        pop(i, :) = idx;
    end
end

function Kids = fisher_mutation(parents,options,nvars,FitnessFcn, state, thisScore,thisPopulation)
    
    Kids = zeros(length(parents),nvars) ;


    for m=1:length(parents)
        parent = thisPopulation(parents(m),:);
        n = length(parent);
        newKid = parent;
        newKid = unique(newKid);
        if length(newKid)<length(parent)
            newKid = [newKid, randi(19824, [1, length(parent)-length(newKid)])];
        end
        for k=1:10
            % randomly select two indices to swap
            i = randi(n);
            newKid(i) = randi(19824);
        end
        Kids(m,:) = newKid ;
    end
end


