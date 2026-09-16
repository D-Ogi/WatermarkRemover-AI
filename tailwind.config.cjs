module.exports = {
            content: ['./ui/**/*.html', './ui/**/*.js', './remwmgui.py'],
            theme: {
                extend: {
                    fontFamily: {
                        mono: ['"Space Mono"', 'monospace'],
                        glitch: ['"Rubik Glitch"', 'cursive'],
                        sans: ['Inter', 'sans-serif'],
                    },
                    colors: {
                        'void': '#050505',
                        'card': '#111116',
                        'neon-green': '#00ff88',
                        'neon-pink': '#ff006e',
                        'neon-cyan': '#00f3ff',
                        'error': '#ff3333',
                    },
                    animation: {
                        'pulse-fast': 'pulse 0.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                        'glitch': 'glitch 1s linear infinite',
                        'marquee': 'marquee 10s linear infinite',
                    },
                    keyframes: {
                        glitch: {
                            '2%, 64%': { transform: 'translate(2px,0) skew(0deg)' },
                            '4%, 60%': { transform: 'translate(-2px,0) skew(0deg)' },
                            '62%': { transform: 'translate(0,0) skew(5deg)' },
                        },
                        marquee: {
                            '0%': { transform: 'translateX(0)' },
                            '100%': { transform: 'translateX(-50%)' },
                        }
                    }
                }
            }
        };
