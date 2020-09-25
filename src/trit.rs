use crate::common::*;

pub trait Trit {
    fn to_bool(&self) -> bool;
}

impl Trit for B1 {
    fn to_bool(&self) -> bool {
        B1::BOOL
    }
}

impl Trit for B0 {
    fn to_bool(&self) -> bool {
        B0::BOOL
    }
}

impl Trit for Unknown {
    fn to_bool(&self) -> bool {
        self.0
    }
}

pub struct Unknown(pub bool);

// and

impl BitAnd<B1> for Unknown {
    type Output = Unknown;

    fn bitand(self, _rhs: B1) -> Self::Output {
        Unknown(self.0.bitand(B1::BOOL))
    }
}

impl BitAnd<B0> for Unknown {
    type Output = B0;

    fn bitand(self, rhs: B0) -> Self::Output {
        rhs
    }
}

impl BitAnd<Unknown> for B1 {
    type Output = Unknown;

    fn bitand(self, rhs: Unknown) -> Self::Output {
        Unknown(B1::BOOL.bitand(rhs.0))
    }
}

impl BitAnd<Unknown> for B0 {
    type Output = B0;

    fn bitand(self, _rhs: Unknown) -> Self::Output {
        self
    }
}

impl BitAnd<Unknown> for Unknown {
    type Output = Unknown;

    fn bitand(self, rhs: Unknown) -> Self::Output {
        Unknown(self.0.bitand(rhs.0))
    }
}

// or

impl BitOr<B1> for Unknown {
    type Output = B1;

    fn bitor(self, rhs: B1) -> Self::Output {
        rhs
    }
}

impl BitOr<B0> for Unknown {
    type Output = Unknown;

    fn bitor(self, _rhs: B0) -> Self::Output {
        Unknown(self.0.bitor(B1::BOOL))
    }
}

impl BitOr<Unknown> for B1 {
    type Output = B1;

    fn bitor(self, _rhs: Unknown) -> Self::Output {
        self
    }
}

impl BitOr<Unknown> for B0 {
    type Output = Unknown;

    fn bitor(self, rhs: Unknown) -> Self::Output {
        Unknown(B0::BOOL.bitor(rhs.0))
    }
}

impl BitOr<Unknown> for Unknown {
    type Output = Unknown;

    fn bitor(self, rhs: Unknown) -> Self::Output {
        Unknown(self.0.bitor(rhs.0))
    }
}

// not

impl Not for Unknown {
    type Output = Unknown;

    fn not(self) -> Self::Output {
        Unknown(self.0.not())
    }
}
