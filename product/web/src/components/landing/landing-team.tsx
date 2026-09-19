import Image from "next/image";

const TEAM = [
  {
    name: "Rubén Godoy",
    role: "CEO",
    src: "/landing/team/ruben.jpg",
    href: "https://www.linkedin.com/in/rubengodoy/",
  },
  {
    name: "Walter J.T.V",
    role: "CTO",
    src: "/landing/team/walter.png",
    href: "https://www.linkedin.com/in/walterjtv/",
  },
  {
    name: "Javier Boix",
    role: "CSO",
    src: "/landing/team/javier.png",
    href: "https://www.linkedin.com/in/javier-boix-campos/",
  },
] as const;

export function LandingTeam() {
  return (
    <div className="flex h-full min-h-0 flex-col justify-center px-8 sm:px-12 lg:px-16 xl:px-20">
      <ul className="grid h-full min-h-0 items-center gap-8 py-8 sm:grid-cols-3 sm:gap-10">
        {TEAM.map((person) => (
          <li key={person.name}>
            <a href={person.href} target="_blank" rel="noopener noreferrer" className="group block">
              <div className="relative aspect-[4/5] overflow-hidden bg-neutral-100">
                <Image
                  src={person.src}
                  alt={person.name}
                  fill
                  sizes="(min-width: 640px) 30vw, 80vw"
                  className="landing-portrait-drift object-cover object-top grayscale transition-[filter] duration-700 group-hover:grayscale-0"
                />
              </div>
              <p className="font-heading mt-4 text-[clamp(1.35rem,2.2vw,1.85rem)] font-semibold leading-none tracking-tight">
                {person.name}
              </p>
              <p className="mt-2 text-[10px] tracking-[0.2em] text-neutral-400 uppercase">{person.role}</p>
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
