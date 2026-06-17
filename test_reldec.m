% Configuration
% load("P_520.mat","P_520")
% % Q = Q ./ max(abs(Q(:)));

% P = P_520;
% blocksize = 10;
% % epsilon_test = 0.05;
% pcmatrix = ldpcQuasiCyclicMatrix(blocksize,P);
% BlockSize = 5;

% P = P_520;
% BlockSize = 10;
% % epsilon_test = 0.05;
% pcmatrix = ldpcQuasiCyclicMatrix(BlockSize,P);
% Q = readmatrix('qtable_ep015000.csv');
% Q = Q(2:end, :);
% pcmatrix = sparse(logical(readmatrix('WRAN_irreg_384_256 (1).csv')));
%% 
load('wran_384_256.mat','wran_384_256');
BlockSize = 16;
H = sparse(logical(wran_384_256));
pcmatrix = H;
blocksize_wran = 16;
load("Q_wran_snr_0.mat","Q")
Q1{1} = Q;
% load("Q_wran_snr_0.5.mat","Q")
% Q1{2} = Q;
% load("Q_wran_snr_1.mat","Q")
% Q1{3} = Q;
% load("Q_wran_snr_1.5.mat","Q")
% Q1{4} = Q;
% load("Q_wran_snr_2.mat","Q")
% Q1{5} = Q;
% load("Q_wran_snr_2.5.mat","Q")
% Q1{6} = Q;
% load("Q_wran_snr_3.mat","Q")
% Q1{7} = Q;
% load("Q_P_520_snr_neg3.mat","Q")
% Q1{1} = Q;
% load("Q_P_520_snr_neg2.mat","Q")
% Q1{2} = Q;
% load("Q_P_520_snr_neg1.mat","Q")
% Q1{3} = Q;
% load("Q_P_520_snr_0.mat","Q")
% Q1{4} = Q;
% load("Q_P_520_snr_1.mat","Q")
% Q1{5} = Q;

[m, ~] = size(pcmatrix);
% P = zeros(m/BlockSize);
CN_neighbors = cell(m,1);
for c = 1:m
    CN_neighbors{c} = find(pcmatrix(c,:));
end
params.maxStateBits = 11;  % 10 for P_520 and % 11 for WRAN
cfgLDPCEnc = ldpcEncoderConfig(pcmatrix);
cfgLDPCDec_bp = ldpcDecoderConfig(pcmatrix);
cfgLDPCDec = ldpcDecoderConfig(pcmatrix,'layered-bp');
% Parameters for Dec
numEdges      = length(cfgLDPCDec.derivedParams.columnIndexMap)/2;
blockLen      = cfgLDPCDec.BlockLength;
parityLen     = cfgLDPCDec.NumParityCheckBits;
nRowsPerLayer = cfgLDPCDec.NumRowsPerLayer;
SNR_db = [0 1 2 3 4];
% SNR_db = [-3 -2 -1 0 1];
% SNR
SNR = 10.^(SNR_db/10);
maxnumiter = 5;

for i = 1 : 1
    current_state = zeros(m,params.maxStateBits);
    i
    parfor j = 1 : 10000
        bits = zeros(cfgLDPCEnc.NumInformationBits,1);
        codeword = ldpcEncode(bits,cfgLDPCEnc);
        codeword_1 = (codeword == 0);
        codeword_2 = (codeword == 1);
        data_modulated = sqrt(SNR(i)).*codeword_1 -sqrt(SNR(i)).*codeword_2;
        noise = randn(cfgLDPCEnc.BlockLength,1);
        data_received = data_modulated + noise;
        soft_demodulated_output = 2*data_received;
        [Y,actualnumiter,finalparitychecks] = ldpcDecode(soft_demodulated_output,cfgLDPCDec,maxnumiter);
        Res = repmat({zeros(cfgLDPCEnc.BlockLength,1)}, 1, height(pcmatrix)/BlockSize);
        Y_temp = soft_demodulated_output;
        % for k = 1 : m
        %     res{k} = zeros(1,numel(CN_neighbors{k}));
        % end

        % --- initialize state ---
        current_state = zeros(m, params.maxStateBits);

        for v = 1:m
            idx1 = CN_neighbors{v};
            vals = Y_temp(idx1);
            vals = vals(:)';

            w = min(length(vals), params.maxStateBits);
            current_state(v,1:w) = vals(1:w);
        end

        % --- hard decision ---
        state_hard_updated = current_state < 0;

        % --- track used clusters ---
        used = false(1, height(pcmatrix)/BlockSize);

        % --- precompute powers (faster) ---
        pow2vec = 2.^(params.maxStateBits-1:-1:0);

        % --- sequential scheduling ---

        for u = 1:(height(pcmatrix)/BlockSize)*maxnumiter   % number of clusters

            % ---------- STATE → INDEX ----------
            s_new = 1 + state_hard_updated * pow2vec';

            % ---------- Q LOOKUP ----------
            vals1 = zeros(1, m);
            for f = 1:m
                vals1(f) = Q1{i}(s_new(f), f);
            end

            % ---------- CLUSTER AGGREGATION ----------
            vals4 = zeros(1, height(pcmatrix)/BlockSize);
            for k = 1:height(pcmatrix)/BlockSize
                start_idx = (k-1)*BlockSize + 1;
                end_idx   = k*BlockSize;
                vals4(k) = sum(vals1(start_idx:end_idx));
            end

            % ---------- MASK USED CLUSTERS ----------
            temp = vals4;
            temp(used) = -inf;

            % ---------- SELECT BEST CLUSTER ----------
            [~, best_cluster] = max(temp);
            % mark cluster as used
            used(best_cluster) = true;

            % ---------- BP UPDATE ----------
            [Y_out, res1] = ldpc_cluster(...
                Y_temp, CN_neighbors, best_cluster, ...
                Res{best_cluster}, BlockSize);

            Y_temp = Y_out;
            Res{best_cluster} = res1;

            % ---------- UPDATE STATE AFTER ACTION ----------
            current_state = zeros(m, params.maxStateBits);

            for v = 1:m
                idx1 = CN_neighbors{v};
                vals = Y_temp(idx1);
                vals = vals(:)';

                w = min(length(vals), params.maxStateBits);
                current_state(v,1:w) = vals(1:w);
            end

            state_hard_updated = current_state < 0;
            if mod(u,height(pcmatrix)/BlockSize) == 0
                used = false(1, height(pcmatrix)/BlockSize);
            end
        end
        output_final_whole = Y_out < 0;
        output_final = output_final_whole(1:cfgLDPCEnc.NumInformationBits);
        ber(j) = biterr(output_final,bits);
        ber1(j) = biterr(Y,bits);
        
    end
    ber_t(i) = mean(ber)/cfgLDPCEnc.NumInformationBits;
    ber_t_1(i) = mean(ber1)/cfgLDPCEnc.NumInformationBits;
end
semilogy(SNR_db,ber_t_1)
semilogy(SNR,ber_t)
grid on