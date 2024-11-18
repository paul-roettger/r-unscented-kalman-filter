use std::ops::{DerefMut, Deref};
use std::fmt;

use crate::matrix_sym as msy;


#[derive(Copy, Clone)]
pub struct MatrixAsym<const M: usize, const N: usize>{
    pub val: [[f64;M];N]
}

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


    pub fn chol_solve(&self, b: &msy::MatrixSym<M>) -> Result<MatrixAsym<M,N>, &'static str>{
        let mut result = MatrixAsym::new();
        let mut m_y = [[0.0;N];M];
        let mut sum;

        let m_l = b.chol()?;

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


#[cfg(test)]
mod tests {
    use crate::masy::MatrixAsym;

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
}
