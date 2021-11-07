use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub struct SimpleError {
    message: String,
}

impl SimpleError {
    pub fn new(message: &'static str) -> SimpleError {
        SimpleError {
            message: message.to_string(),
        }
    }

    pub fn from_string(message: String) -> SimpleError {
        SimpleError { message }
    }
}

impl Display for SimpleError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for SimpleError {}
