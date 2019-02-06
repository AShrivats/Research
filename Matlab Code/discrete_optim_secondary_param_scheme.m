% Discrete time solver
% allowing eta, psi to be non-zero
tic
T = 1; % length of period
pen = 300; % penalty per unit of non compliance
S_min = 0;
S_max = pen;
req = 500; % requirement for compliance
b_max = 2*req;
h = 1*req; % constant for now, can make this deterministic in time without anything changing
mu_f = 0; % drift of F_t (assumed ABM here, but more realistically a jump process)
sigma_f = 10; % volatility of F_t (assumed ABM here, but more realistically a jump process)
zeta = 0.6;
gamma = 0.6;
time_steps = 50;
dt = T/time_steps;
dS = sqrt(3 * dt) * sigma_f;
grid_points_s = ceil((S_max - S_min)/dS);
grid_points_b = 101; % have to choose this carefully so that we get enough grid points for the interpolation to not be problematic
T_grid = linspace(0, 1, time_steps+1);
psi = 1/200;
eta = 1/100;
nsim = 200;

options = optimset('Display', 'off', 'MaxFunEvals', 100); % options for minimization
s = rng; % set seed
% initialization of parameters
S_grid = linspace(S_min, S_max, grid_points_s);
b_grid = linspace(0, 2*req, grid_points_b);
[bb, ss] = meshgrid(b_grid, S_grid);
pars = [time_steps pen req h mu_f sigma_f zeta gamma T psi eta];

V = zeros(time_steps, grid_points_s, grid_points_b);
gen_opt = zeros(time_steps, grid_points_s, grid_points_b);
trade_opt = zeros(time_steps, grid_points_s, grid_points_b);
% iterating through timesteps and minimizing the cost at each step
for t = time_steps:-1:1
    s_noise = normrnd(0, sqrt(dt), [1, nsim]);
    for s = 1:grid_points_s
        for b = 1:grid_points_b
            if b_grid(b) + h * dt * (time_steps-t+1) >= req % guaranteed to comply with baseline generation
                b0 = h;
                t0 = -S_grid(s)/gamma; 
            elseif b_grid(b) + (h + pen / zeta) * dt * (time_steps-t+1) <= req
                % not going to comply even with extreme production
                b0 = h + pen / zeta;
                t0 = (pen - S_grid(s))/gamma;
                if t0 == 0 % helps with getting the right minimum
                    t0 = -1;
                end
            else % should roughly correspond to the 'interesting' regime
                if s > 1
                    b0 = gen_opt(t, s-1, b);
                    t0 = trade_opt(t, s-1, b)-1;% -1 to make sure it doesnt get stuck at 0
                elseif b>1
                    b0 = gen_opt(t, s, b-1);
                    t0 = trade_opt(t, s, b-1)-1;% -1 to make sure it doesnt get stuck at 0
                else
                    b0 = gen_opt(t+1, 1, 1);
                    t0 = trade_opt(t+1, 1, 1); % if we're at (1,1), look at the future period
                end
            end
            x0 = [b0 t0];
            %y = runningCost([b0-40 t0-40], t, S_grid(s), b_grid(b),pars, V, b_grid, S_grid, s_noise)
            f = @(ctrl)runningCost(ctrl, t, S_grid(s), b_grid(b), pars, V, b_grid, S_grid, s_noise);
            [x, fval] = fminsearch(f, x0, options);
            V(t, s, b) = fval;
            gen_opt(t,s,b) = x(1);
            trade_opt(t,s,b) = x(2);
        end
          s
    end
    t
end
toc
save('single_period_secondary_pars.mat')
% %load('single_period_t1.mat')
% tt = time_steps-1;%time_steps-10;
% f1 = figure('visible','on');
% surf(bb, ss, squeeze(V(tt,:,:)), 'edgecolor', 'none')
% ylabel('SREC Price')
% xlabel('Banked SRECs')
% zlabel(strcat('Value function at t = ', num2str(T_grid(tt))))
% title(strcat('Value function at t = ', num2str(T_grid(tt))))
% 
% f2 = figure('visible','on');
% surf(bb, ss, squeeze(gen_opt(tt,:,:)), 'edgecolor','none')
% ylabel('SREC Price')
% xlabel('Banked SRECs')
% zlabel(strcat('Optimal generation at t = ', num2str(T_grid(tt))))
% title(strcat('Optimal generation t = ', num2str(T_grid(tt))))
% 
% f3 = figure('visible','on');
% surf(bb, ss, squeeze(trade_opt(tt,:,:)), 'edgecolor', 'none')
% ylabel('SREC Price')
% xlabel('Banked SRECs')
% zlabel(strcat('Optimal trading at t = ', num2str(T_grid(tt))))
% title(strcat('Optimal trading t = ', num2str(T_grid(tt))))

%y = runningCost([528.1498 -8.1498], 52, S_grid(10), b_grid(50), pars, V, b_grid, S_grid, s_noise)
%y = runningCost([528.0105 -8.0061], 52, S_grid(10), b_grid(50), pars, V, b_grid, S_grid, s_noise)
function y = runningCost(ctrl, t, S, b, pars, vals, b_grid, S_grid, s_noise)
% actions, timestep, stock price, banking level, model parameters, old
% values, grid_points
time_steps = pars(1);
pen = pars(2);
req = pars(3);
h = pars(4);
mu_f = pars(5);
sigma_f = pars(6);
zeta = pars(7);
gamma = pars(8);
T = pars(9);
psi = pars(10);
eta = pars(11);

dt = T/time_steps;
gen = ctrl(1);
trade= ctrl(2);
    if t == time_steps
        y = 1 / 2 * zeta * (gen - h)^2 *dt + trade.*S*dt +...
            1 / 2 * gamma * trade^2 *dt + pen*max(req - (gen + trade)*dt - b, 0);
    else
        new_b = min(max(0, (b + (gen + trade)*dt)), 2*req);
        new_S = max(0, min(pen, S + mu_f * dt - psi * gen * dt + eta * trade * dt + sigma_f * s_noise));
        running_value = 1 / 2 * zeta * (gen - h)^2 *dt + trade*S*dt +...
            1 / 2 * gamma * trade^2 *dt;
        V = squeeze(vals(t+1, : ,:));
        [X, Y] = meshgrid(b_grid, S_grid);
        Vq = interp2(X,Y,V,new_b, new_S);
        future_value = mean(Vq); % need to interpolate this from vals(t+1,:,:)
        y = running_value + future_value;        
    end
end

