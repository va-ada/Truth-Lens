import { motion } from 'framer-motion';
import { forwardRef } from 'react';

const Button = forwardRef(function Button(
  { variant = 'primary', as = 'button', href, children, onClick, className = '', ...rest },
  ref,
) {
  const cls = `btn btn-${variant} ${className}`;
  const Motion = motion[as] || motion.button;
  const motionProps = {
    whileHover: { scale: 1.02 },
    whileTap: { scale: 0.97 },
    transition: { type: 'spring', stiffness: 400, damping: 28 },
  };
  if (as === 'a') {
    return (
      <Motion ref={ref} href={href} className={cls} onClick={onClick} {...motionProps} {...rest}>
        {children}
      </Motion>
    );
  }
  return (
    <Motion ref={ref} type="button" className={cls} onClick={onClick} {...motionProps} {...rest}>
      {children}
    </Motion>
  );
});

export default Button;
