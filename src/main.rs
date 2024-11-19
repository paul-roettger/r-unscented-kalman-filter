mod matrix_asym;
mod matrix_sym;

use crate::matrix_asym as masy;
use crate::matrix_sym as msy;

fn main() {
    println!("Hello, world!");
}

/* struct to hold the data for an unscented kalman filter */
struct UKF<const L: usize, const M: usize, const N: usize> {
    /* internal scaling factor */
    lambda: f64,

    /* System function to simulate the system state: x(t) = f_sys(x(t-t_sample)) */
    f_sys: fn(&masy::MatrixAsym<1,L>, &masy::MatrixAsym<1,N>) -> masy::MatrixAsym<1,L>,
    /* System funktion to calculate the system output given the system state: y(t) = h_sys(x(t)) */
    h_sys: fn(&masy::MatrixAsym<1,L>) -> masy::MatrixAsym<1,M>,

    /* internal weights factors */
    wmc0: f64,
    wm0: f64,
    wma: f64,

    /* Covariance of the process noise */
    q: msy::MatrixSym<L>,

    /* Covariance of the measurement noise */
    r: msy::MatrixSym<M>,

    /* Covariance of the filter error */
    p: msy::MatrixSym<L>,

    /* Filtered system state */
    x_p: masy::MatrixAsym<1,L>
}

impl<const L: usize, const M: usize, const N: usize> UKF<L, M, N>{
    pub fn new(
               /* scaling factor for spread of sigma points. Set to value between 1e-4 and 1 */
               alpha: f64, 
               /* scaling factor for distribution. For Gaussian set o 2 */
               beta: f64, 
               /* scaling factor, is usually set to 0*/
               kappa: f64, 
               /* system functions */
               f_sys: fn(&masy::MatrixAsym<1,L>, &masy::MatrixAsym<1,N>) -> masy::MatrixAsym<1,L>,
               h_sys: fn(&masy::MatrixAsym<1,L>) -> masy::MatrixAsym<1,M>,
               /* initial value of the state */
               x_start: &masy::MatrixAsym<1,L>,
               /* initial value of the covariance of the filter error */
               p_start: &msy::MatrixSym<L>, 
               /* Covariance of the process noise */
               q: &msy::MatrixSym<L>, 
               /* Covariance of the measurement noise */
               r: &msy::MatrixSym<M>) -> Self {

        let lambda = (alpha*alpha*(L as f64 + kappa)) - L as f64;

        Self { 
            lambda,
            f_sys,
            h_sys,
            wm0: lambda/(lambda + L as f64),
            wmc0: lambda/(lambda + L as f64) + 1.0 - alpha*alpha + beta,
            wma: 1.0/(2.0*(lambda + L as f64)),
            q: *q,
            r: *r,
            p: *p_start,
            x_p: *x_start
        }
    }

    /* Update the filtered system state with a new measurement and system input */
    pub fn update(&mut self, yt: &masy::MatrixAsym<1,M>, ut: &masy::MatrixAsym<1,N>) -> Result<(), &'static str>{

        // Step 1 Generate Sigma points
        let (chi_p_b, chi_p_c) = {
            let mut s_p = self.p.chol()?;
            s_p.scalar_prod((self.lambda + L as f64).sqrt());

            let mut chi_p_b = [masy::MatrixAsym::new(); L];
            let mut chi_p_c = [masy::MatrixAsym::new(); L];
            for j in 0..L
            {
                let mut temp_in: masy::MatrixAsym<1, L> = masy::MatrixAsym::new();
                for i in 0..L{
                    temp_in[i][0] = (*s_p)[i][j];
                }
                chi_p_b[j] = *self.x_p.clone().add(&temp_in);
                chi_p_c[j] = *self.x_p.clone().sub(&temp_in);
            }

            (chi_p_b, chi_p_c)
        };

        //Step 2: Prediction of Transformation
        let (chi_m_a, chi_m_b, chi_m_c, x_m) = {

            let chi_m_a = (self.f_sys)(&self.x_p, ut);
            let mut chi_m_b: [masy::MatrixAsym<1, L> ;L] = [masy::MatrixAsym::new(); L];
            let mut chi_m_c: [masy::MatrixAsym<1, L> ;L] = [masy::MatrixAsym::new(); L];

            let mut x_m = chi_m_a.clone();
            x_m.scalar_prod(self.wm0);

            for ((c_m_b, c_m_c), 
                (c_p_b, c_p_c)) 
                    in chi_m_b.iter_mut().zip(chi_m_c.iter_mut())
                                        .zip(chi_p_b.iter().zip(chi_p_c.iter())){
                *c_m_b = (self.f_sys)(c_p_b, ut);
                *c_m_c = (self.f_sys)(c_p_c, ut);
                x_m.add(c_m_b.clone().scalar_prod(self.wma));
                x_m.add(c_m_c.clone().scalar_prod(self.wma));
            }
            (chi_m_a, chi_m_b, chi_m_c, x_m)
        };

        let p_m = {

            let mut p_m = self.q.clone();

            p_m.add(&(chi_m_a.clone().sub(&x_m)
                            .self_mult_selft::<L>())
                            .scalar_prod(self.wmc0));

            for (c_m_b, c_m_c) in chi_m_b.iter().zip(chi_m_c.iter()){
                p_m.add((*c_m_b).clone()
                                .sub(&x_m)
                                .self_mult_selft::<L>()
                                .scalar_prod(self.wma));

                p_m.add((*c_m_c).clone()
                                .sub(&x_m)
                                .self_mult_selft::<L>()
                                .scalar_prod(self.wma));
            }

            p_m
        };
        

        // Step 3: Observation transformation
        let (psi_m_a, psi_m_b, psi_m_c, y_m) = {
            let psi_m_a ;
            let mut psi_m_b = [masy::MatrixAsym::new(); L];
            let mut psi_m_c = [masy::MatrixAsym::new(); L];
            let mut y_m = masy::MatrixAsym::new();

            psi_m_a = (self.h_sys)(&chi_m_a);
            y_m.add(psi_m_a.clone().scalar_prod(self.wm0));

            for ((c_m_b, c_m_c), 
                (p_m_b, p_m_c)) in 
                        chi_m_b.iter().zip(chi_m_c.iter()).zip(psi_m_b.iter_mut().zip(psi_m_c.iter_mut())){
                *p_m_b = (self.h_sys)(c_m_b);
                *p_m_c = (self.h_sys)(c_m_c);
                y_m.add((*p_m_b).clone().scalar_prod(self.wma));
                y_m.add((*p_m_c).clone().scalar_prod(self.wma));
            }
            (psi_m_a, psi_m_b, psi_m_c, y_m)
        };

        let (p_yy, p_xy) = {
            let mut p_yy = self.r.clone();
            let mut p_xy =  masy::MatrixAsym::new();

            p_yy.add(psi_m_a.clone()
                            .sub(&y_m)
                            .self_mult_selft::<M>()
                            .scalar_prod(self.wmc0));
            p_xy.add(&chi_m_a.clone()
                            .sub(&x_m)
                            .scalar_prod(self.wmc0)
                            .mult(&psi_m_a.clone()
                                            .sub(&y_m)
                                            .transpose()));
            for ((c_m_b, c_m_c), 
                (p_m_b, p_m_c)) in 
                chi_m_b.iter().zip(chi_m_c.iter()).zip(psi_m_b.iter().zip(psi_m_c.iter())){
                p_yy.add((*p_m_b).clone()
                                    .sub(&y_m)
                                    .self_mult_selft::<M>()
                                    .scalar_prod(self.wma));
                p_xy.add(&(*c_m_b).clone()
                                    .sub(&x_m)
                                    .scalar_prod(self.wma)
                                    .mult(&(*p_m_b).clone()
                                                    .sub(&y_m)
                                                    .transpose()));

                p_yy.add((*p_m_c).clone()
                                    .sub(&y_m)
                                    .self_mult_selft::<M>()
                                    .scalar_prod(self.wma));
                p_xy.add(&(*c_m_c).clone()
                                    .sub(&x_m)
                                    .scalar_prod(self.wma)
                                    .mult(&(*p_m_c).clone()
                                                    .sub(&y_m)
                                                    .transpose()));
            }

            (p_yy, p_xy)
        };

        // Step 4. Measurement update
        let k = p_xy.chol_solve(&p_yy)?;
        self.x_p = *k.mult(&yt.clone().sub(&y_m)).add(&x_m);
        self.p = *p_m.clone().sub(&p_yy.b_mult_self_mult_bt(&k));

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use crate::{masy::MatrixAsym, msy::MatrixSym, UKF};
    use rand::distributions::Distribution;
    use csv::Writer;

    /* System functions for a simple non-linear system to be observed */
    fn ukf_test_f_sys(x: &MatrixAsym<1,4>, _u: &MatrixAsym<1,0>) -> MatrixAsym<1,4>
    {
        let t = 0.1;
        let mut f_sys = MatrixAsym::new();
        *f_sys = [[1.0, 0.0,   t, 0.0],
                  [0.0, 1.0, 0.0,   t],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]];
        f_sys.mult(x)
    }

    fn ukf_test_h_sys(x: &MatrixAsym<1,4>) -> MatrixAsym<1,2>
    {
        let n1 = 20.0;
        let n2 = 0.0;
        let e1 = 0.0; 
        let e2 = 20.0;
        let mut result = MatrixAsym::new();
        result[0][0] = ((x[0][0] - n1).powi(2) + (x[1][0] - e1).powi(2)).sqrt();
        result[1][0] = ((x[0][0] - n2).powi(2) + (x[1][0] - e2).powi(2)).sqrt();
        result
    }

    #[test]
    fn ukf_test() {
        let t_samplesize = 500;

        /* Generate matrices */
        let mut x_start = MatrixAsym::new();
        *x_start  = [[0.0],
                     [0.0],
                     [50.0],
                     [50.0]];
        let mut p_start = MatrixSym::new();
        *p_start = [[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]];

        let mut q = MatrixSym::new();
        *q = [[0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 4.0, 0.0],
              [0.0, 0.0, 0.0, 4.0]];

        let mut r = MatrixSym::new();
        *r = [[1.0, 0.0],
              [0.0, 1.0]];

        /* Calculate squareroot of covariance matrices*/
        let mut sqrt_q = q.0.clone();
        let mut sqrt_r = r.0.clone();
        let mut sqrt_p = p_start.0.clone();
        for i in 0..4{
            for j in 0..4{
                if i == j{
                    sqrt_q[i][j] = sqrt_q[i][j].sqrt();
                }
                else {
                    sqrt_q[i][j] = 0.0;
                }
            }
        }
        for i in 0..2{
            for j in 0..2{
                if i == j{
                    sqrt_r[i][j] = sqrt_r[i][j].sqrt();
                }
                else {
                    sqrt_r[i][j] = 0.0;
                }
            }
        }
        for i in 0..4{
            for j in 0..4{
                if i == j{
                    sqrt_p[i][j] = sqrt_p[i][j].sqrt();
                }
                else {
                    sqrt_p[i][j] = 0.0;
                }
            }
        }

        /* Generate standard distributed noise 
           w_t: Process noise
           v_t: Measurement noise*/
        let distr = rand::distributions::Standard;
        let w_t = distr.sample_iter(rand::thread_rng())
                                              .take(t_samplesize)
                                              .map(|(w1, w2, w3, w4)| {
                                                  let mut v = MatrixAsym::new();
                                                  *v = [[w1],
                                                        [w2],
                                                        [w3],
                                                        [w4]];
                                                  sqrt_q.mult(&v)
                                              })
                                              .collect::<Vec<_>>();

        let v_t = distr.sample_iter(rand::thread_rng())
                                              .take(t_samplesize)
                                              .map(|(v1, v2)| {
                                                  let mut v = MatrixAsym::new();
                                                  *v = [[v1],
                                                        [v2]];
                                                  sqrt_r.mult(&v)
                                              })
                                              .collect::<Vec<_>>();

        /* Generate the initial value for the real state of the system */
        let mut x_t = distr.sample_iter(rand::thread_rng())
                                                  .take(1)
                                                  .map(|(v1, v2, v3, v4)| {
                                                      let mut v = MatrixAsym::new();
                                                      *v = [[v1],
                                                           [v2],
                                                           [v3],
                                                           [v4]];
                                                      *sqrt_p.mult(&v)
                                                             .add(&x_start)
                                                  })
                                                  .collect::<Vec<_>>();
        
        /* Generate the system sate and add process noise */
        let u = MatrixAsym::new();
        for i in 0..t_samplesize-1{
            x_t.push(*ukf_test_f_sys(&x_t[i], &u).add(&w_t[i]))
        }

        /* Generate the measurement output based on the system state and add measurement noise */
        let mut y_t = vec![];
        for (x, v) in x_t.iter().zip(v_t.iter()){
            y_t.push(*ukf_test_h_sys(x).add(v));
        }

        /* Begin the filtering */
        let mut ukf = UKF::new(1.0,
                                              2.0,
                                             0.0, 
                                                    ukf_test_f_sys, 
                                                    ukf_test_h_sys, 
                                                    &x_start, 
                                                    &p_start, 
                                                    &q, 
                                                    &r);
        
        /* Update the observed system state using measurement y and input u */
        let mut x_ukf = vec![];
        for y in y_t.iter(){
            x_ukf.push(ukf.x_p);
            ukf.update(y, &u).unwrap();
        }

        /* Write the true and the estimated system state in a csv file  */
        let mut wrt = Writer::from_path("system_state.csv").unwrap();
        wrt.write_record(["x1_p", "x2_p", "x3_p", "x4_p", "x1", "x2", "x3", "x4",]).unwrap();
        for (x_test_ukf, x) in x_ukf.iter().zip(x_t.iter()){
            wrt.write_record(x_test_ukf.iter().chain(x.iter()).map(|v| v[0].to_string())).unwrap();
        }
        wrt.flush().unwrap();
    }

}