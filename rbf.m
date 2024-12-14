n = length(Y);
c = cvpartition(n, 'HoldOut', 0.2); % creating a cvpartition object with holding %20 of the data for validation
X_normal = normalize(X,1);
i_nan = any(isnan(X_normal),1);
X_normal(:, i_nan) = [];
Index_names = get_index_names(59);
Index_names(i_nan) = [];

Xtst_normal = normalize(Xtst,1);
i_nan = any(isnan(Xtst_normal),1);
Xtst_normal(:, i_nan) = [];

figure
plot(X_normal(Y == 1, 1), X_normal(Y == 1, 2), 'r.', 'MarkerSize', 10); % plotting the data of class 1 with red color
hold on; % holding the current plot
plot(X_normal(Y == -1, 1), X_normal(Y == -1, 2), 'b.', 'MarkerSize', 10); % plotting the data of class 2 with blue color
hold off; % releasing the current plot
title('The data of two classes in two-dimensional space'); % adding a title
xlabel('Feature 1'); % adding a label for the horizontal axis
ylabel('Feature 2'); % adding a label for the vertical axis
legend('Class 0', 'Class 1'); % adding a legend
saveas(gcf, 'Q1.png')

%%
n_features = 40;
c = cvpartition(n,'KFold',5);


accuracy_best = 0;
spread_best = 0;
n_best = 0;
n_folds = 5;
net_best = [];
for n_neuron=25:2:35
    for spread=6:0.2:10
        %n_neuron = 35;
        %spread = 5.6;
        acc = 0;
        for k=1:5
            id_train = training(c,k);
            id_test = test(c,k);
            Xtrain = X_normal(id_train, Index(1:n_features)); % training data
            Ytrain = round(Y(id_train)/2+0.5); % training labels
            Xtest = X_normal(id_test, Index(1:n_features)); % validation data
            Ytest = round(Y(id_test)/2+0.5); % validation labels

            %n_neuron = 20;
            %spread = 3;
            net = newrb(Xtrain', Ytrain', 0.0, spread, n_neuron, n_neuron); % designing a network with one neuron in the output
            
            Ypred = round(net(Xtest')); % predict the labels for the validation data
            acc = acc + mean(Ypred == Ytest')/n_folds; % calculate the accuracy
            
        end
        disp(acc)
        if acc>accuracy_best
            accuracy_best = acc;
            spread_best = spread;
            n_best = n_neuron;
            net_best = net;
        end
    end
end

fprintf('number of neurons: %d, spread: %d, highest  accuray: %d', n_best, spread_best, accuracy_best)
TestLabel_rbf = round(net_best(Xtst_normal(:, Index(1:n_features))'));
%%
net = newrb(X_normal(:, Index(1:40))', Y', 0.0, 5.6, 35, 35); % designing a network with one neuron in the output
Xtst_normal = normalize(Xtst,1);
i_nan = any(isnan(Xtst_normal),1);
Xtst_normal(:, i_nan) = [];
TestLabel_rbf = round(net(Xtst_normal(:, Index(1:40))')); % predict the labels for the validation data

%%

%%
c = cvpartition(n,'KFold',5);


accuracy_best = 0;
spread_best = 0;
n_best = 0;
n_folds = 5;
n_features = 60;
net_best = [];
for n_neuron=11:1:16
    for spread=11:0.2:13
        %n_neuron = 35;
        %spread = 5.6;
        acc = 0;
        for k=1:5
            id_train = training(c,k);
            id_test = test(c,k);
            Xtrain = X_normal(id_train, solution(1:n_features)); % training data
            Ytrain = round(Y(id_train)/2+0.5); % training labels
            Xtest = X_normal(id_test, solution(1:n_features)); % validation data
            Ytest = round(Y(id_test)/2+0.5); % validation labels

            %n_neuron = 20;
            %spread = 3;
            net = newrb(Xtrain', Ytrain', 0.0, spread, n_neuron, n_neuron); % designing a network with one neuron in the output
            
            Ypred = round(net(Xtest')); % predict the labels for the validation data
            acc = acc + mean(Ypred == Ytest')/n_folds; % calculate the accuracy
            
        end
        disp(acc)
        if acc>accuracy_best
            accuracy_best = acc;
            spread_best = spread;
            n_best = n_neuron;
            net_best = net;
        end
    end
end

fprintf('number of neurons: %d, spread: %d, highest  accuray: %d', n_best, spread_best, accuracy_best)

%net = newrb(X_normal(:, solution(1:n_features))', Y', 0.0, 12, 16, 16); % designing a network with one neuron in the output

TestLabel_rbf_GA = round(net_best(Xtst_normal(:, solution(1:n_features))')); % predict the labels for the validation data

