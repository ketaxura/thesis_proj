import casadi as ca


def create_random_solver(robotpos, T=0.1, N=10, max_obs=4, max_static=4):
    x,y,theta,  = ca.SX.sym('x'), ca.SX.sym('y', ca.SX.sym('theta'))
    states      = ca.vertcat(x,y,theta)
    v,omega     = ca.SX.sym('v'), ca.SX.sym('omega')
    controls    = ca.vertcat(v,omega)
    rhs         = ca.vertcat(v*ca.cos(theta), v*ca.sin(theta),omega)
    f           = ca.Function('f',[states,controls], [rhs])
    
    
    U= ca.SX.sym('U', 2, N)
    X= ca.SX.sym('X', 3, N+1)
    P= ca.SX.sym('P',
        3               #initial state
        +2*N            #waypoints
        +N              #headings
        +2*N*max_obs    #moving_obs
        +2*max_static   #static obs
        +5              #weights
    )
    
    Q_goal      =P[-5]
    Q_smooth    =P[-4]
    Q_obs       =P[-3]
    Q_speed     =P[-2]
    Q_theta     =P[-1]
    
    
    
    
    
    

    