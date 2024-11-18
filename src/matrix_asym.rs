use std::ops::{DerefMut, Deref};
use std::fmt;

use crate::matrix_sym as msy;

/* Assymmetric matrix of generic size */
#[derive(Copy, Clone)]
pub struct MatrixAsym<const M: usize, const N: usize>{
    pub val: [[f64;M];N]
}

impl<const M: usize, const N: usize> MatrixAsym<M,N>{

    pub fn new() -> Self{
        Self { val: [[0.0; M];N] }
    }

    /* add two matrices of the same size */
    pub fn add(&mut self, b: &MatrixAsym<M,N>) -> &mut Self{
        for (linea, lineb) in self.deref_mut().iter_mut().zip(b.deref().iter()){
            for (a,b) in linea.iter_mut().zip(lineb.iter()){
                *a += *b;
            }
        }
        self
    }

    /* subtract two matrices of the same size */
    pub fn sub(&mut self, b: &MatrixAsym<M,N>) -> &mut Self{
        for (linea, lineb) in self.deref_mut().iter_mut().zip(b.deref().iter()){
            for (a,b) in linea.iter_mut().zip(lineb.iter()){
                *a -= *b;
            }
        }
        self
    }

    /* multiplicate two matrices the matrix dimensions may change */
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

    /* Perform the matrix operation M*(M^t) 
       This operation gets a seperate function because mathematically it alway returns a symmetric matrix.
       If M.mult(&transpose(&M)) was used, the result could be asymmetric due to rounding errors.*/
    pub fn self_mult_selft<const U: usize>(&self) -> msy::MatrixSym<N>{

        let mut result = msy::MatrixSym::new();
        let mut sum;
        for i in 0..N {
            for j in 0..=i {
                sum = 0.0;
                for k in 0..M {
                    sum += self[i][k] * self[j][k];
                }
                result[i][j] = sum;
                result[j][i] = sum;
            }
        }
        result
    }

    /* Calculate the scalar product */
    pub fn scalar_prod(&mut self, b: f64) -> &mut Self{
        for line in self.deref_mut().iter_mut(){
            for value in line.iter_mut(){
                *value *= b;
            }
        }
        self
    }

    /* Get the transpose to a given matrix */
    pub fn transpose(&self) -> MatrixAsym<N,M>{
        let mut result = MatrixAsym::new();
        for i in 0..M {
            for j in 0..N {
                result[i][j] = self[j][i];
            }
        }
        result
    }

    /* Solve the equation x*A = b using Cholesky decomposition
       A has to be symmetric and positive definite */
    pub fn chol_solve(&self, a: &msy::MatrixSym<M>) -> Result<MatrixAsym<M,N>, &'static str>{
        let mut result = MatrixAsym::new();
        let mut m_y = [[0.0;N];M];
        let mut sum;

        /* Perform Cholesky decomposition */
        let m_l = a.chol()?;

        /* forward substitution */
        for i in 0..N{
            for j in 0..M{
                sum = 0.0;
                for k in 0..j{
                    sum += m_y[k][i] * m_l[j][k];
                }
                if m_l[j][j] != 0.0{
                    m_y[j][i] = (self[i][j]-sum)/m_l[j][j];
                }else{
                    return Err("Div 0");
                }
            }
        }

        /* back substitution */
        for i in 0..N{
            for j in (0..M).rev(){
                sum = 0.0;
                for k in (j..M).rev(){
                    sum += result[i][k] * m_l[k][j];
                }
                if m_l[j][j] != 0.0{
                    result[i][j] = (m_y[j][i]-sum)/m_l[j][j];
                }else{
                    return Err("Div 0");
                }
            }
        }

        Ok(result)
    }

    /* Check if two matrices are similar
       If one or more entries differs between the matrices by a value > tolerance returns false, else return true */
    pub fn similar(&self, b: &MatrixAsym<M,N>, tolerance: f64) -> bool{
        for (linea, lineb) in self.deref().iter().zip(b.deref().iter()){
            for (a,b) in linea.iter().zip(lineb.iter()){
                if (a - b).abs() > tolerance{
                    return false;
                }
            }
        }
        true
    }

}


/* Implementing deref to access val more easily */
impl<const M: usize, const N: usize> Deref for MatrixAsym<M,N>
{
    type Target = [[f64;M];N];

    fn deref(&self) -> &Self::Target{
        &self.val
    }
}

/* Implementing derefmut to access val more easily */
impl<const M: usize, const N: usize> DerefMut for MatrixAsym<M,N>
{
    fn deref_mut(&mut self) -> &mut Self::Target{
        &mut self.val
    }
}

/* Implementing Display to show results in tests */
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


#[cfg(test)]
mod tests {
    use crate::{masy::MatrixAsym, msy::MatrixSym};

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

        let mut d = MatrixAsym::new();
        *d = [[0.0,0.0],
              [0.0,0.0],
              [0.0,0.0],
              [0.0,0.0]];

        assert!(c.similar(&d, 0.001)); 
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

        let mut d = MatrixAsym::new();
        *d = [[7.0,8.0],
              [9.0,2.0],
              [7.0,8.0],
              [9.0,2.0]];

        print!("{}\n*{}\n={}",a,b,c);    

        assert!(c.similar(&d, 0.001));  
 
    }    

    #[test]
    fn matrix_solve() {

        let mut a = MatrixSym::new();
        *a =   [[2.0, 1.0, 0.5],
                [1.0, 9.0, 0.1],
                [0.5, 0.1, 3.0]];

        let mut b = MatrixAsym::new();
        *b =   [[1.0, 1.0, 1.0]];

        /* Solve the equation x*A = b */
        let x = b.chol_solve(&a).unwrap();

        /* Check if x*A == b */
        let d = x.mult(&a.0);
        assert!(d.similar(&b, 0.001));
    }

    #[test]
    fn matrix_inv() {

        let mut a = MatrixSym::new();
        *a =   [[2.0, 1.0, 0.5],
                [1.0, 9.0, 0.1],
                [0.5, 0.1, 3.0]];

        let mut i = MatrixAsym::new();
        *i =   [[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]];

        let b = i.chol_solve(&a).unwrap();

        /* Check if (A⁻1)*A == I */
        assert!(b.mult(&a.0).similar(&i, 0.001));

    }
}
