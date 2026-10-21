function [LLR_out1,Res_int] = ldpc_cluster(LLR_in,C,a,Res,BlockSize)

Res_int = zeros(length(LLR_in),1);
LLR_int = LLR_in;
LLR_int = LLR_int - Res;
for j = 1 : BlockSize
    idx = C{(a-1)*BlockSize+j};
    llr_temp = LLR_int(idx);
    x = abs(llr_temp);
    sgn = sign(llr_temp);

    % Stable phi transformation
    phi_x = log1p(2 ./ expm1(x));

    % Total sign product
    sign_prod = prod(sgn);

    % Compute each outgoing message
    sum_phi = sum(phi_x);
    sum_phi_excl = sum_phi - phi_x;

    % Stable inverse phi transformation
    mag = log1p(2 ./ expm1(sum_phi_excl));

    % Handle numerical extremes
    mag(sum_phi_excl == 0) = 50;
    mag(~isfinite(mag)) = 50;

    % Outgoing check-node messages
    Res_int(idx) = sign_prod .* sgn .* min(mag, 50);
end
LLR_out1 = LLR_int + Res_int;
end