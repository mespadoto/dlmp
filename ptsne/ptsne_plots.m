pkg load statistics

% Load MNIST dataset
load 'mnist_train.mat';
load 'mnist_test.mat';

X = [train_X; test_X];
y = [train_labels; test_labels];

% Set perplexity and network structure
perplexity = 30;
layers = [500 500 2000 2];

test_sizes = [2000, 10000, 30000, 60000, 100000];
train = randsample(1:70000, 5000);

% Train the parametric t-SNE network
disp('Start training');
t = tic;
[network, err] = train_par_tsne(X(train,:), y(train,:), X(train,:), y(train,:), layers, 'CD1');
train_elapsed = toc(t);
disp(['Training elapsed time: ', num2str(train_elapsed)]);

for test_size = test_sizes
  test  = randsample(1:70000, test_size, Replace=true);
  
  t = tic;
  mapped_test_X  = run_data_through_network(network, X(test,:));
  test_elapsed = toc(t);
  disp([num2str(test_size),',', num2str(test_elapsed), ',', num2str(test_elapsed + train_elapsed)]);

  csvwrite(strcat('X_ptsne_', num2str(test_size), '.csv'), mapped_test_X);
  csvwrite(strcat('y_ptsne_', num2str(test_size), '.csv'), y(test,:));
endfor;
