export default function Link({ to, className = '', children, ...rest }) {
  return <a href={`#${to}`} className={className} {...rest}>{children}</a>
}
