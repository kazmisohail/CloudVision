import { Linkedin, Mail, Github } from 'lucide-react';
import Image from 'next/image';

export default function TeamPage() {
    return (
        <main className="min-h-screen bg-[#0f172a] overflow-x-hidden relative">
            {/* Background Elements */}
            <div className="fixed inset-0 z-0">
                <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop')] bg-cover bg-center opacity-50"></div>
                <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[#0f172a]/50 to-[#0f172a]"></div>
            </div>

            <div className="relative z-10 w-full max-w-6xl mx-auto px-4 py-24 space-y-16">
                <div className="text-center space-y-4">
                    <h1 className="text-4xl md:text-5xl font-bold text-white">Meet the Team</h1>
                    <p className="text-cyan-400 max-w-2xl mx-auto">
                        The minds behind CloudVision.
                    </p>
                </div>

                {/* Supervisor Card */}
                <div className="flex justify-center">
                    <div className="w-full max-w-md bg-gradient-to-b from-slate-800/80 to-slate-900/80 backdrop-blur-md border-2 border-yellow-500/50 rounded-2xl p-1 shadow-2xl shadow-yellow-500/10">
                        <div className="bg-slate-900/90 rounded-xl p-8 text-center h-full relative overflow-hidden">
                            <div className="absolute top-0 right-0 bg-yellow-500 text-black text-xs font-bold px-3 py-1 rounded-bl-lg">
                                SUPERVISOR
                            </div>
                            <div className="w-32 h-32 mx-auto bg-slate-700 rounded-full mb-6 border-4 border-slate-800 relative overflow-hidden">
                                <Image src="/team/supervisor.jpeg" alt="Supervisor" fill className="object-cover" />
                            </div>
                            <h3 className="text-2xl font-bold text-white">Ms. Amna Iftikhar</h3>
                            <p className="text-yellow-400 font-medium mb-2">Final Year Project Supervisor</p>
                            <p className="text-gray-400 text-sm mb-6">Senior Lecturer at Bahria University</p>
                            <div className="flex justify-center gap-4">
                                <a href="mailto:supervisor@example.com" className="p-2 bg-slate-800 rounded-full hover:bg-slate-700 text-gray-400 hover:text-white transition-colors">
                                    <Mail className="h-5 w-5" />
                                </a>
                                <a href="https://linkedin.com/" target="_blank" rel="noopener noreferrer" className="p-2 bg-slate-800 rounded-full hover:bg-slate-700 text-gray-400 hover:text-white transition-colors">
                                    <Linkedin className="h-5 w-5" />
                                </a>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Students Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {[
                        { name: "Sohail Abbas", role: "Student", color: "cyan", img: "sohail.jpg", github: "https://github.com/kazmisohail", linkedin: "https://linkedin.com/in/kazmisohail", email: "02-134221-025@student.bahria.edu.pk" },
                        { name: "Sofia Haider", role: "Student", color: "blue", img: "sofia.jpeg", github: "https://github.com/", linkedin: "https://linkedin.com/", email: "student@example.com" },
                        { name: "Emad Tariq", role: "Student", color: "purple", img: "emad.jpeg", github: "https://github.com/", linkedin: "https://linkedin.com/", email: "student@example.com" }
                    ].map((member, index) => (
                        <div key={index} className="bg-slate-800/50 backdrop-blur-md border border-white/10 rounded-xl p-6 hover:bg-slate-800/70 transition-all hover:-translate-y-1 duration-300 shadow-lg">
                            <div className="w-24 h-24 mx-auto bg-slate-700 rounded-full mb-4 relative overflow-hidden">
                                <Image src={`/team/${member.img}`} alt={member.name} fill className="object-cover" />
                            </div>
                            <div className="text-center">
                                <h3 className="text-xl font-semibold text-white">{member.name}</h3>
                                <p className={`text-${member.color}-400 text-sm font-medium mb-4`}>{member.role}</p>
                                <div className="flex justify-center gap-3">
                                    <a href={member.github} target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-white transition-colors"><Github className="h-4 w-4" /></a>
                                    <a href={member.linkedin} target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-white transition-colors"><Linkedin className="h-4 w-4" /></a>
                                    <a href={`mailto:${member.email}`} className="text-gray-400 hover:text-white transition-colors"><Mail className="h-4 w-4" /></a>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </main>
    );
}
