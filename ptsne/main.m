pkg load statistics

% Load MNIST dataset
load 'mnist_train.mat';
load 'mnist_test.mat';

X = [train_X; test_X];
y = [train_labels; test_labels];

% Set perplexity and network structure
perplexity = 30;
layers = [500 500 2000 2];

test_sizes = [1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 75000, 100000, 500000, 1000000];
train = randsample(1:70000, 5000);

% Train the parametric t-SNE network
disp('Start training');
t = tic;
[network, err] = train_par_tsne(X(train,:), y(train,:), X(train,:), y(train,:), layers, 'CD1');
train_elapsed = toc(t);
disp(['Training elapsed time: ', num2str(train_elapsed)]);

times = zeros(length(test_sizes), 3)

for i = 1:length(test_sizes)
	test_size = test_sizes(i)
	test  = randsample(1:70000, test_size, Replace=true);
  
	t = tic;
	mapped_test_X  = run_data_through_network(network, X(test,:));
	test_elapsed = toc(t);

	disp([num2str(test_size),',', num2str(test_elapsed), ',', num2str(test_elapsed + train_elapsed)]);
	times(i,:) = [test_size, test_elapsed, test_elapsed + train_elapsed];

	if test_size <= 100000
		disp(strcat('Saving X_ptsne_',num2str(test_size), '.csv'));
		csvwrite(strcat('../data/X_ptsne_', num2str(test_size), '.csv'), mapped_test_X);
		csvwrite(strcat('../data/X_test_ptsne_', num2str(test_size), '.csv'), X(test,:));
		csvwrite(strcat('../data/y_ptsne_', num2str(test_size), '.csv'), y(test,:));
	endif
endfor;

csvwrite('../data/ptsne_times.csv', times);
