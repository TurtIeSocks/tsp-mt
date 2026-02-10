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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::SpecWriter;

    #[test]
    fn writer_emits_expected_lines_for_key_value_helpers() {
        struct Render;
        impl std::fmt::Display for Render {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut w = SpecWriter::new(f);
                w.line("HEAD")?;
                w.lines("VALUES", &[1, 2])?;
                w.lines::<i32>("EMPTY", &[])?;
                w.kv_eq("A", 10)?;
                w.opt_kv_eq("B", None::<u8>)?;
                w.opt_kv_eq("B", Some(11))?;
                w.kv_colon("C", 12)?;
                w.opt_kv_colon("D", None::<u8>)?;
                w.opt_kv_colon("D", Some(13))?;
                w.opt_path_eq("P", Some(&PathBuf::from("problem.par")))?;
                w.path_list_eq("CANDIDATE_FILE", &[PathBuf::from("a"), PathBuf::from("b")])?;
                Ok(())
            }
        }

        let out = format!("{}", Render);
        assert_eq!(
            out,
            "HEAD\nVALUES\n1\n2\nA = 10\nB = 11\nC: 12\nD: 13\nP = problem.par\nCANDIDATE_FILE = a\nCANDIDATE_FILE = b\n"
        );
    }

    #[test]
    fn row_skips_empty_and_formats_space_separated_values() {
        struct Render;
        impl std::fmt::Display for Render {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut w = SpecWriter::new(f);
                w.row::<i32>(&[])?;
                w.row(&[7, 8, 9])?;
                Ok(())
            }
        }

        assert_eq!(format!("{}", Render), "7 8 9\n");
    }
}
