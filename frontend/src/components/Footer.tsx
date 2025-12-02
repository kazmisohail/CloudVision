import { Github } from 'lucide-react';

export default function Footer() {
    return (
        <footer className="bg-slate-900/50 backdrop-blur-lg border-t border-white/10 shadow mt-auto">
            <div className="w-full max-w-screen-xl mx-auto p-4 md:py-8">
                <span className="block text-sm text-gray-400 sm:text-center">
                    © 2025{' '}
                    <a href="https://github.com/kazmisohail/cloudyyvision" className="hover:underline">
                        CloudyyVision™
                    </a>
                    . All Rights Reserved.
                </span>
            </div>
        </footer>
    );
}
