use std::{
    fmt::{Display, Formatter},
    ops::{Deref, DerefMut},
    path::PathBuf,
};

pub(crate) struct SpecWriter<'a, 'b>(&'a mut Formatter<'b>);

impl<'a, 'b> Deref for SpecWriter<'a, 'b> {
    type Target = Formatter<'b>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a, 'b> DerefMut for SpecWriter<'a, 'b> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}

impl<'a, 'b> SpecWriter<'a, 'b> {
    pub(crate) fn new(f: &'a mut Formatter<'b>) -> Self {
        Self(f)
    }

    pub(crate) fn line<T: Display>(&mut self, value: T) -> std::fmt::Result {
        writeln!(self, "{value}")
    }

    pub(crate) fn lines<T: Display>(&mut self, key: &str, values: &[T]) -> std::fmt::Result {
        if values.is_empty() {
            return Ok(());
        }
        self.line(key)?;
        for val in values {
            self.line(val)?;
        }
        Ok(())
    }

    pub(crate) fn kv_eq<T: Display>(&mut self, key: &str, value: T) -> std::fmt::Result {
        writeln!(self, "{key} = {value}")
    }

    pub(crate) fn opt_kv_eq<T: Display>(
        &mut self,
        key: &str,
        value: Option<T>,
    ) -> std::fmt::Result {
        if let Some(value) = value {
            self.kv_eq(key, value)?;
        }
        Ok(())
    }

    pub(crate) fn kv_colon<T: Display>(&mut self, key: &str, value: T) -> std::fmt::Result {
        writeln!(self, "{key}: {value}")
    }

    pub(crate) fn opt_kv_colon<T: Display>(
        &mut self,
        key: &str,
        value: Option<T>,
    ) -> std::fmt::Result {
        if let Some(value) = value {
            self.kv_colon(key, value)?;
        }
        Ok(())
    }

    pub(crate) fn opt_path_eq(&mut self, key: &str, value: Option<&PathBuf>) -> std::fmt::Result {
        if let Some(value) = value {
            self.kv_eq(key, value.display())?;
        }
        Ok(())
    }

    pub(crate) fn path_list_eq(&mut self, key: &str, values: &[PathBuf]) -> std::fmt::Result {
        for value in values {
            self.kv_eq(key, value.display())?;
        }
        Ok(())
    }

    pub(crate) fn row<T: Display>(&mut self, row: &[T]) -> std::fmt::Result {
        if row.is_empty() {
            return Ok(());
        }

        write!(self, "{}", row[0])?;
        for value in &row[1..] {
            write!(self, " {value}")?;
        }
        self.line("")
    }
}
