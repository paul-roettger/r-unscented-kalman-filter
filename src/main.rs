use std::ops::{DerefMut, Deref};
use std::{fmt, vec};
use std::convert::From;
use futures::executor::block_on;

fn main() {
    println!("Hello, world!");
}

#[derive(Copy, Clone)]
struct MatrixAsym<const M: usize, const N: usize>{
    pub val: [[f64;M];N]
}

#[derive(Copy, Clone)]
struct MatrixSym<const M: usize>(MatrixAsym<M,M>);

impl<const M: usize, const N: usize> MatrixAsym<M,N>{

    pub fn new() -> Self{
        Self { val: [[0.0; M];N] }
    }

    pub fn add(&mut self, b: &MatrixAsym<M,N>) -> &mut Self{
        for (linea, lineb) in self.deref_mut().iter_mut().zip(b.deref().iter()){
            for (a,b) in linea.iter_mut().zip(lineb.iter()){
                *a += *b;
            }
        }
        self
    }

    pub fn sub(&mut self, b: &MatrixAsym<M,N>) -> &mut Self{
        for (linea, lineb) in self.deref_mut().iter_mut().zip(b.deref().iter()){
            for (a,b) in linea.iter_mut().zip(lineb.iter()){
                *a -= *b;
            }
        }
        self
    }

    pub fn mult<const U: usize>(&self, b: &MatrixAsym<U,M>) -> MatrixAsym<U,N>{

        let mut result = MatrixAsym::new();
        let mut sum;
        for i in 0..N {
            for j in 0..U {
                sum = 0.0;
                for k in 0..M {
                    sum += self[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        result
    }

    pub fn self_mult_selft<const U: usize>(&self) -> MatrixSym<N>{

        let a: MatrixAsym<M,N> = MatrixAsym::new();
        let b: MatrixAsym<N,M> = MatrixAsym::new();
        let c = a.mult(&b);

        let mut result = MatrixSym::new();
        let mut sum;
        for i in 0..N {
            for j in 0..=i {
                sum = 0.0;
                for k in 0..M {
                    sum += self[i][k] * b[j][k];
                }
                result[i][j] = sum;
                result[j][i] = sum;
            }
        }
        result
    }

    pub fn scalar_prod(&mut self, b: f64) -> &mut Self{
        for line in self.deref_mut().iter_mut(){
            for value in line.iter_mut(){
                *value *= b;
            }
        }
        self
    }

    pub fn transpose(&self) -> MatrixAsym<N,M>{
        let mut result = MatrixAsym::new();
        for i in 0..M {
            for j in 0..N {
                result[i][j] = self[j][i];
            }
        }
        result
    }

}

impl<const M: usize, const N: usize> Deref for MatrixAsym<M,N>
{
    type Target = [[f64;M];N];

    fn deref(&self) -> &Self::Target{
        &self.val
    }
}

impl<const M: usize, const N: usize> DerefMut for MatrixAsym<M,N>
{
    fn deref_mut(&mut self) -> &mut Self::Target{
        &mut self.val
    }
}


impl<const M: usize, const N: usize> fmt::Display for MatrixAsym<M,N>{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[\n")?;
        for line in self.deref(){
            write!(f, "[")?;
            for item in line{
                write!(f, "{},", *item)?;
            }
            write!(f, "]\n")?;
        }
        write!(f, "]")
    }
}


impl<const M: usize> MatrixSym<M>{
    pub fn new() -> Self{
        Self { 0: MatrixAsym::new() }
    }

    pub fn b_mult_self_mult_bt<const U: usize>(&self, b: &MatrixAsym<M,U>) -> MatrixSym<U>{
        let mut result = MatrixSym::new();
        let mut tmp = [[0.0; M]; U];
        let mut sum;
        for i in 0..U{
            for j in 0..M {
                sum = 0.0;
                for k in 0..M {
                    sum += b[i][k] * self[k][j];
                }
                tmp[i][j] = sum;
            }
        }
        for i in 0..U {
            for j in 0..=i {
                sum = 0.0;
                for k in 0..M {
                    sum += tmp[i][k] * b[j][k];
                }
                result[i][j] = sum;
                result[j][i] = sum;
            }
        }
        result
    }

    pub fn chol(&self) -> Result<MatrixAsym<M,M>, &'static str>{
        let mut sum;
        let mut result = MatrixAsym::new();

        /* Cholesky decomposition */
        for i in 0..M{
            for j in 0..=i{

                sum = 0.0;
                for k in 0..j{
                    sum += result[i][k] * result[j][k];
                }

                if i == j{
                    if self[i][i] - sum >= 0.0 {
                        result[i][j] = (self[i][i] - sum).sqrt();
                    }else {
                        return Err("Negative squareroot");
                    }
                } else {
                    if result[j][j] != 0.0{
                        result[i][j] = (1.0 / result[j][j]) * (self[i][j] - sum);
                    }else{
                        return Err("Div 0");
                    }
                }
            }
        }
        Ok(result)
    }

    pub fn chol_solve<const U: usize>(&self, b: &MatrixAsym<U,M>) -> Result<MatrixAsym<U,M>, &'static str>{
        let mut result = MatrixAsym::new();
        let mut m_y = [[0.0;U];M];
        let mut sum;

        let mut m_l = self.chol()?;

        /* forward substitution */
        for i in 0..U{
            for j in 0..M{
                sum = 0.0;
                for k in 0..j{
                    sum += m_y[k][i] * m_l[j][k];
                }
                if m_l[j][j] != 0.0{
                    m_y[j][i] = (b[j][i]-sum)/m_l[j][j];
                }else{
                    return Err("Div 0");
                }
            }
        }

        /* back substitution */
        for i in 0..U{
            for j in (0..M).rev(){
                sum = 0.0;
                for k in (j..M).rev(){
                    sum += result[k][i] * m_l[k][j];
                }
                if m_l[j][j] != 0.0{
                    result[j][i] = (m_y[j][i]-sum)/m_l[j][j];
                }else{
                    return Err("Div 0");
                }
            }
        }
        Ok(result)
    }


}

impl<const M: usize> Deref for MatrixSym<M>
{
    type Target = [[f64;M];M];

    fn deref(&self) -> &Self::Target{
        &self.0.deref()
    }
}

impl<const M: usize> DerefMut for MatrixSym<M>
{
    fn deref_mut(&mut self) -> &mut Self::Target{
        let reference = self.0.deref_mut();
        reference
    }
}

impl<const M: usize> fmt::Display for MatrixSym<M>{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
} 


struct UKF<const L: usize, const M: usize, const N: usize> {
    pub alpha: f64,
    pub beta: f64,
    pub kappa: f64,
    pub lambda: f64,
    f_sys: fn(&MatrixAsym<1,L>, &MatrixAsym<1,N>) -> MatrixAsym<1,L>,
    h_sys: fn(&MatrixAsym<1,L>) -> MatrixAsym<1,M>,
    wmc0: f64,
    wm0: f64,
    wma: f64,
    q: MatrixSym<L>,
    r: MatrixSym<M>,
    p: MatrixSym<L>,
    x_p: MatrixAsym<1,L>
}

impl<const L: usize, const M: usize, const N: usize> UKF<L, M, N>{
    pub fn new(alpha: f64, 
        beta: f64, 
        kappa: f64, 
        f_sys: fn(&MatrixAsym<1,L>, &MatrixAsym<1,N>,) -> MatrixAsym<1,L>,
        h_sys: fn(&MatrixAsym<1,L>) -> MatrixAsym<1,M>,
        q: &MatrixSym<L>, 
        r: &MatrixSym<M>) -> Self {

        let lambda = alpha*alpha*(L as f64 + kappa) - L as f64;

        let mut p = MatrixSym::new();
        for i in 0..L
        {
            p[i][i] = f64::MAX/2.0;
        }

        Self { 
            alpha,
            beta,
            kappa,
            lambda,
            f_sys,
            h_sys,
            wm0: lambda/(lambda + L as f64),
            wmc0: lambda/(lambda + L as f64) + 1.0 - alpha*alpha + beta,
            wma: 1.0/(2.0*(lambda + L as f64)),
            q: *q,
            r: *r,
            p,
            x_p: MatrixAsym::new()
        }
    }


    pub fn update(&mut self, yt: &MatrixAsym<1,M>, ut: &MatrixAsym<1,N>) -> Result<(), &'static str>{

        let mut s_p = self.p.chol()?;
        s_p.scalar_prod((self.lambda + L as f64).sqrt());

        let mut ones = MatrixSym::new();
        for i in 0..L{
            for j in 0..L{
                ones[j][i] = self.x_p[0][i];
            }
        }
        let mut chi_p_b = ones.0.clone();
        let mut chi_p_c = ones.0.clone();
        chi_p_b.add(&s_p);
        chi_p_c.sub(&s_p);

        let mut chi_m_a = (self.f_sys)(&self.x_p, ut);
        let mut chi_m_b: [MatrixAsym<1, L> ;L] = [MatrixAsym::new(); L];
        let mut chi_m_c: [MatrixAsym<1, L> ;L] = [MatrixAsym::new(); L];

        let mut x_m = chi_m_a.clone();
        x_m.scalar_prod(self.wm0);

        for j in 0..L{
            let mut  temp_in: MatrixAsym<1, L> = MatrixAsym::new();
            for i in 0..L{
                temp_in[0][i] = (*chi_p_b)[i][j];
            }
            let mut temp_out = (self.f_sys)(&temp_in, ut);
            chi_m_b[j] = temp_out;

            x_m.add(temp_out.scalar_prod(self.wma));
        }

        for j in 0..L{
            let mut  temp_in: MatrixAsym<1, L> = MatrixAsym::new();
            for i in 0..L{
                temp_in[0][i] = (*chi_p_c)[i][j];
            }
            let mut temp_out= (self.f_sys)(&temp_in, ut);
            chi_m_b[j] = temp_out;

            x_m.add(temp_out.scalar_prod(self.wma));
        }

        let mut p_m = self.q.clone();

        p_m.0.add(&(chi_m_a.sub(&x_m)
                           .self_mult_selft::<L>().0)
                           .scalar_prod(self.wm0));

        for j in 0..L{
            p_m.0.add(&(chi_m_b[j].sub(&x_m)
                                  .self_mult_selft::<L>().0)
                                  .scalar_prod(self.wma));

            p_m.0.add(&(chi_m_c[j].sub(&x_m)
                                  .self_mult_selft::<L>().0)
                                  .scalar_prod(self.wma));
        }

        let mut psi_m_a = MatrixAsym::new();
        let mut psi_m_b = [MatrixAsym::new(); L];
        let mut psi_m_c = [MatrixAsym::new(); L];
        let mut y_m = MatrixAsym::new();

        psi_m_a = (self.h_sys)(&chi_m_a);
        y_m.add(psi_m_a.scalar_prod(self.wm0));

        for i in 0..L{
            psi_m_b[i] = (self.h_sys)(&chi_m_b[i]);
            psi_m_c[i] = (self.h_sys)(&chi_m_c[i]);
            y_m.add(psi_m_b[i].scalar_prod(self.wm0));
            y_m.add(psi_m_c[i].scalar_prod(self.wm0));
        }

        let mut p_yy = self.r.clone();
        let mut p_xy =  MatrixAsym::new();

        p_yy.0.add(psi_m_a.clone()
                           .sub(&y_m)
                           .self_mult_selft::<M>().0
                           .scalar_prod(self.wmc0));

        let a = psi_m_a.clone().sub(&y_m).transpose();
        let b = chi_m_a.clone().sub(&x_m).transpose();
        let c = a.mult(b);
        p_xy.add(chi_m_a.clone().sub(&x_m).mult(&psi_m_a.clone().sub(&y_m).transpose()));

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use crate::{MatrixAsym, MatrixSym};

    #[test]
    fn add_test() {
        let mut a = MatrixAsym::new();
        *a = [[1.0,2.0],
              [3.0,4.0],
              [5.0,6.0],
              [7.0,8.0]];

        let mut b = MatrixAsym::new();
        *b = [[-1.0,-2.0],
              [-3.0,-4.0],
              [-5.0,-6.0],
              [-7.0,-8.0]];

        let c = a.add(&b);
        print!("{}",*c);     
 
    }

    #[test]
    fn mul_test() {
        let mut a = MatrixAsym::new();
        *a = [[3.0,2.0,1.0],
              [1.0,0.0,2.0],
              [3.0,2.0,1.0],
              [1.0,0.0,2.0]];

        let mut b = MatrixAsym::new();
        *b = [[1.0,2.0],
              [0.0,1.0],
              [4.0,0.0]];

        let c = a.mult(&b);

        print!("{}\n*{}\n={}",a,b,c);     
 
    }    

    #[test]
    fn normal_mult_t() {
        let mut a = MatrixSym::new();
        *a =   [[1.0,2.0,1.0,2.0],
                [2.0,4.0,1.0,2.0],
                [1.0,1.0,1.0,2.0],
                [2.0,2.0,2.0,2.0]];

        let mut b = MatrixAsym::new();
        *b =   [[0.0,1.0,2.0,3.0],
                [4.0,3.0,2.0,1.0]];

        let mut c = MatrixAsym::new();
        *c =   [[0.0,4.0],
                [1.0,3.0],
                [2.0,2.0],
                [3.0,1.0]];

        let d = a.b_mult_self_mult_bt(&b);
        print!("{}\n",d); 

        let tmp = b.mult(&(a.0));
        let e = tmp.mult(&c);
        print!("{}\n",e); 
 
    }

    #[test]
    fn matrix_inv() {
        let mut a = MatrixSym::new();
        *a =   [[2.0, 1.0],
                [1.0,9.0]];

        let mut b = MatrixAsym::new();
        *b =   [[1.0, 0.0],
                [0.0, 1.0]];

        let b = a.chol_solve(&b).unwrap();
        print!("{}\n",b); 

    }

}